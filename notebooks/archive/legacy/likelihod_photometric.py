#!/usr/bin/env python
# coding: utf-8

import logging
import re
import time
from collections.abc import Sequence
from copy import copy, deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from nautilus import Prior, Sampler

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import photo_cov as pcov
from cosmicfishpie.LSSsurvey import photo_obs as pobs
from cosmicfishpie.utilities.utils import printing as upr


def is_indexable_iterable(var):
    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


logger = logging.getLogger("cosmicfishpie.cosmology.nuisance")
logger.setLevel(logging.INFO)


upr.debug = False


upr.debug_print("test")

outfolder = "nautichain_results"
outroot = "cosmicjellyfish_Euclid-ISTF-Pess-3x2photo_symb_withnuis"

outpath = Path(outfolder)
outpath.mkdir(parents=True, exist_ok=True)  # parents=True creates parent directories if needed
outfilepath = outpath / f"{outroot}"
outfilepath = str(outfilepath)
print(f"Results will be saved to {outfilepath}")

sampler_settings = {
    "n_live": 2000,
    "n_networks": 16,
    "n_batch": 256,
    "pool": 64,
}


fiducial = {
    "Omegam": 0.3145714273,
    "Omegab": 0.0491989,
    "h": 0.6737,
    "ns": 0.96605,
    "sigma8": 0.81,
    "w0": -1.0,
    "wa": 0.0,
    "mnu": 0.06,
    "Neff": 3.044,
}
observables = ["WL", "GCph"]


options = {
    "accuracy": 1,
    "feedback": 1,
    "code": "symbolic",
    "outroot": outroot,
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
    "specs_dir": "./cosmicfishpie/configs/default_survey_specifications/",
    # relative to where script is launched from not where it is located
    "cosmo_model": "LCDM",
}
cosmoFM_fid = cosmicfish.FisherMatrix(
    fiducialpars=fiducial,
    options=options,
    observables=observables,
    cosmoModel=options["cosmo_model"],
    surveyName=options["survey_name"],
)


cosmoFM_fid.IApars


photo_fid = pobs.ComputeCls(
    cosmopars=cosmoFM_fid.fiducialcosmopars,
    photopars=cosmoFM_fid.photopars,
    IApars=cosmoFM_fid.IApars,
    biaspars=cosmoFM_fid.photobiaspars,
)

photo_fid.compute_all()

photo_cov_fid = pcov.PhotoCov(
    cosmopars=cosmoFM_fid.fiducialcosmopars,
    photopars=cosmoFM_fid.photopars,
    IApars=cosmoFM_fid.IApars,
    biaspars=cosmoFM_fid.photobiaspars,
    fiducial_Cls=photo_fid,
)


photo_cov_fid.allparsfid


print(photo_fid.binrange_WL)
print(photo_fid.binrange_GCph)
print(photo_cov_fid.ngalbin_WL)
print(photo_cov_fid.ngalbin_GCph)


def observable_Cell(photo_th: pobs.ComputeCls):
    photo_th.compute_all()
    binrange_GCph = photo_th.binrange_GCph
    binrange_WL = photo_th.binrange_WL
    nbin_GCph = len(binrange_GCph)
    nbin_WL = len(binrange_WL)

    ells = photo_th.result["ells"]
    output = dict(ells=ells)

    observables = photo_th.observables
    if "WL" in observables:
        Cell_LL = np.empty((len(ells), nbin_WL, nbin_WL), dtype=np.float64)
    if "GCph" in observables:
        Cell_GG = np.empty((len(ells), nbin_GCph, nbin_GCph), dtype=np.float64)
    if "WL" in observables and "GCph" in observables:
        Cell_GL = np.empty((len(ells), nbin_GCph, nbin_WL), dtype=np.float64)

    for i, j in product(binrange_WL, binrange_GCph):

        if "WL" in observables:
            Cell_LL[:, i - 1, j - 1] = (
                photo_th.result["WL {}xWL {}".format(i, j)]
                + np.eye(nbin_WL)[i - 1, j - 1]
                * photo_cov_fid.ellipt_error**2.0
                / photo_cov_fid.ngalbin_WL[i - 1]
            )

        if "GCph" in observables:
            Cell_GG[:, i - 1, j - 1] = (
                photo_th.result["GCph {}xGCph {}".format(i, j)]
                + np.eye(nbin_GCph)[i - 1, j - 1] * 1 / photo_cov_fid.ngalbin_GCph[i - 1]
            )

        if "WL" in observables and "GCph" in observables:
            Cell_GL[:, i - 1, j - 1] = photo_th.result["GCph {}xWL {}".format(i, j)]

    if "WL" in observables:
        output["Cell_LL"] = Cell_LL
    if "GCph" in observables:
        output["Cell_GG"] = Cell_GG
    if "WL" in observables and "GCph" in observables:
        output["Cell_GL"] = Cell_GL

    return output


Cells_fid = observable_Cell(photo_fid)
print(Cells_fid.keys())


print(Cells_fid["Cell_LL"].shape)
print(Cells_fid["Cell_GG"].shape)
print(Cells_fid["Cell_GL"].shape)


ellmax_WL = cosmoFM_fid.specs["lmax_WL"]
ellmax_GC = cosmoFM_fid.specs["lmax_GCph"]
ellmax_XC = np.minimum(ellmax_GC, ellmax_WL)
nbins_Glob = min(len(list(photo_fid.binrange_WL)), len(list(photo_fid.binrange_GCph)))
print(nbins_Glob)


def compute_chi2_per_obs(Cell_fid, Cell_th, ells, dells):

    dfid = np.linalg.det(Cell_fid)
    dth = np.linalg.det(Cell_th)

    nells = len(ells)
    _, _, nbin = Cell_fid.shape

    dmix = 0
    for i in range(nbin):
        Cth_mix = copy(Cell_th)
        Cth_mix[:, i, :] = Cell_fid[:, i, :]
        dmix += np.linalg.det(Cth_mix)

    ingrd = (2 * ells + 1) * (
        dmix[:nells] / dth[:nells] + np.log(dth[:nells] / dfid[:nells]) - nbin
    )
    ingrd = [*((ingrd[1:] + ingrd[:-1]) / 2 * dells[:-1]), ingrd[-1] * dells[-1]]

    chi2 = np.sum(ingrd)
    return chi2


def compute_chi2(Cells_fid, Cells_th):
    """
    Compute χ² for wedges using fully vectorized operations.
    Matches the loop implementation exactly.

    Parameters:
    ----------
    Cells_fid: Dict

    Cells_th: Dict

    Returns:
    -------
    float
        χ² value
    """
    chi2 = 0
    ells = Cells_fid["ells"]

    if "WL" in observables and "GCph" not in observables:
        Cells_WL_th = Cells_th["Cell_LL"]
        Cells_WL_fid = Cells_fid["Cell_LL"]

        iWL = np.searchsorted(ells, ellmax_WL)
        ells_WL = np.insert(ells, iWL, ellmax_WL)
        Dl_WL = np.diff(ells_WL)[:iWL]
        ells_WL = ells_WL[:iWL]

        chi2 += photo_cov_fid.fsky_WL * compute_chi2_per_obs(
            Cells_WL_fid, Cells_WL_th, ells_WL, Dl_WL
        )

    if "GCph" in observables and "WL" not in observables:
        Cells_GC_th = Cells_th["Cell_GG"]
        Cells_GC_fid = Cells_fid["Cell_GG"]

        iGC = np.searchsorted(ells, ellmax_GC)
        ells_GC = np.insert(ells, iGC, ellmax_GC)
        Dl_GC = np.diff(ells_GC)[:iGC]
        ells_GC = ells_GC[:iGC]

        chi2 += photo_cov_fid.fsky_GCph * compute_chi2_per_obs(
            Cells_GC_fid, Cells_GC_th, ells_GC, Dl_GC
        )

    if "GCph" in observables and "WL" in observables:
        Cells_XC_th = Cells_th["Cell_GL"]
        Cells_XC_fid = Cells_fid["Cell_GL"]
        Cells_GC_th = Cells_th["Cell_GG"]
        Cells_GC_fid = Cells_fid["Cell_GG"]
        Cells_WL_th = Cells_th["Cell_LL"]
        Cells_WL_fid = Cells_fid["Cell_LL"]

        iGC = np.searchsorted(ells, ellmax_GC)
        ells_GC = np.insert(ells, iGC, ellmax_GC)
        Dl_GC = np.diff(ells_GC)[:iGC]
        ells_GC = ells_GC[:iGC]
        iWL = np.searchsorted(ells, ellmax_WL)
        ells_WL = np.insert(ells, iWL, ellmax_WL)
        Dl_WL = np.diff(ells_WL)[:iWL]
        ells_WL = ells_WL[:iWL]
        iXC = np.searchsorted(ells, ellmax_XC)
        ells_XC = np.insert(ells, iXC, ellmax_XC)
        Dl_XC = np.diff(ells_XC)[:iXC]
        ells_XC = ells_GC[:iXC]

        big_th = np.block(
            [
                [Cells_WL_th[:iXC], np.transpose(Cells_XC_th, (0, 2, 1))[:iXC]],
                [Cells_XC_th[:iXC], Cells_GC_th[:iXC]],
            ]
        )
        big_fid = np.block(
            [
                [Cells_WL_fid[:iXC], np.transpose(Cells_XC_fid, (0, 2, 1))[:iXC]],
                [Cells_XC_fid[:iXC], Cells_GC_fid[:iXC]],
            ]
        )

        chi2 += np.sqrt(photo_cov_fid.fsky_WL * photo_cov_fid.fsky_GCph) * compute_chi2_per_obs(
            big_fid, big_th, ells_XC, Dl_XC
        )
        chi2 += photo_cov_fid.fsky_WL * compute_chi2_per_obs(
            Cells_WL_fid[:iXC], Cells_WL_th[:iXC], ells_WL[:iXC], Dl_WL[:iXC]
        )

    return chi2


def loglike(param_vec, prior=None):

    if isinstance(param_vec, dict):
        param_dict = deepcopy(param_vec)
    elif is_indexable_iterable(param_vec) and prior is not None:
        # print(f'Loading prior with keys: {prior.keys}')
        param_dict = {key: param_vec[i] for i, key in enumerate(prior.keys)}

    photopars = deepcopy(cosmoFM_fid.photopars)
    for ii, pp in enumerate(cosmoFM_fid.photopars.keys()):
        photopars[ii] = param_dict.pop(pp, cosmoFM_fid.photopars[pp])

    photobiaspars = deepcopy(cosmoFM_fid.photobiaspars)
    for ii, pp in enumerate(cosmoFM_fid.photobiaspars.keys()):
        photobiaspars[pp] = param_dict.pop(pp, cosmoFM_fid.photobiaspars[pp])

    IApars = deepcopy(cosmoFM_fid.IApars)
    for ii, pp in enumerate(cosmoFM_fid.IApars.keys()):
        IApars[pp] = param_dict.pop(pp, cosmoFM_fid.IApars[pp])

    photo_vary = pobs.ComputeCls(
        param_dict,
        photopars,
        IApars,
        photobiaspars,
    )
    Cells_th = observable_Cell(photo_vary)

    return -0.5 * compute_chi2(Cells_fid, Cells_th)


cosmoFM_fid.freeparams


cosmoFM_fid.allparams


samp1dic = {
    "Omegam": 0.3145714273,
    "Omegab": 0.0491989,
    "h": 0.6737,
    "ns": 0.96605,
    "sigma8": 0.81,
    "w0": -1.0,
    "wa": 0.0,
    "mnu": 0.06,
    "Neff": 3.044,
    "bias_model": "binned",
    "b1": 1.0997727037892875,
    "b2": 1.220245876862528,
    "b3": 1.2723993083933989,
    "b4": 1.316624471897739,
    "b5": 1.35812370570578,
    "b6": 1.3998214171814918,
    "b7": 1.4446452851824907,
    "b8": 1.4964959071110084,
    "b9": 1.5652475842498528,
    "b10": 1.7429859437184225,
    "fout": 0.1,
    "co": 1,
    "cb": 1,
    "sigma_o": 0.05,
    "sigma_b": 0.05,
    "zo": 0.1,
    "zb": 0.0,
    "IA_model": "eNLA",
    "AIA": 1.72,
    "betaIA": 2.17,
    "etaIA": -0.41 * 1.1,
}
print("Sample likelihood", loglike(samp1dic))


loglike(photo_cov_fid.allparsfid)


photo_cov_fid.allparsfid


fishmat_photo = cosmoFM_fid.compute()


prior_nonuis = Prior()
prior_withnuis = Prior()


prior_dict = {
    "Omegam": [0.24, 0.4],
    "Omegab": [0.04, 0.06],
    "h": [0.61, 0.75],
    "ns": [0.92, 1.00],
    "sigma8": [0.79, 0.83],
    "AIA": [1.0, 3.0],
    "etaIA": [-6.0, 6.0],
    "b1": [1.0, 3.0],
    "b2": [1.0, 3.0],
    "b3": [1.0, 3.0],
    "b4": [1.0, 3.0],
    "b5": [1.0, 3.0],
    "b6": [1.0, 3.0],
    "b7": [1.0, 3.0],
    "b8": [1.0, 3.0],
    "b9": [1.0, 3.0],
    "b10": [1.0, 3.0],
}


cosmoFM_fid.freeparams


for par in prior_dict.keys():
    if par in cosmoFM_fid.freeparams.keys():
        dist_prior = (prior_dict[par][0], prior_dict[par][1])
        if re.match(r"b\d+", par):
            prior_withnuis.add_parameter(par, dist_prior)
        elif re.search(r"IA", par):
            prior_withnuis.add_parameter(par, dist_prior)
        else:
            prior_nonuis.add_parameter(par, dist_prior)
            prior_withnuis.add_parameter(par, dist_prior)


print(prior_nonuis.keys)
print(prior_withnuis.keys)


if "withnuis" in options["outroot"]:
    prior_chosen = prior_withnuis
elif "nonuis" in options["outroot"]:
    prior_chosen = prior_nonuis
else:
    raise ValueError("No prior specified in the outroot")
print("Loading prior with keys: ", prior_chosen.keys)


tini = time.time()
print("Starting sampler at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tini)))


sampler = Sampler(
    prior_chosen,
    loglike,
    n_live=sampler_settings["n_live"],
    n_networks=sampler_settings["n_networks"],
    n_batch=sampler_settings["n_batch"],
    pool=sampler_settings["pool"],
    pass_dict=False,
    filepath=outfilepath + ".hdf5",
    resume=True,
    likelihood_kwargs={"prior": prior_chosen},
)
sampler.run(verbose=True, discard_exploration=True)
log_z_all = sampler.evidence()
print("Evidence:", log_z_all)
points_all, log_w_all, log_l_all = sampler.posterior()


tfin = time.time()
elapsed = tfin - tini
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
print("Sampler finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tfin)))
print(f"Total time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")


sample_wghlkl = np.vstack((points_all.T, np.exp(log_w_all), log_l_all)).T


outfile_chain = outfilepath + ".txt"
print(f"Saving chain to text file {outfile_chain}")


headerlist = ["loglike", "weights"] + list(prior_chosen.keys)
header = " ".join(headerlist)
print("Saving header: ", header)


np.savetxt(outfile_chain, sample_wghlkl, header=header)
print("Finished Sampling Run")
