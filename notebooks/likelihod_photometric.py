#!/usr/bin/env python
# coding: utf-8
import logging
import re
from collections.abc import Sequence
from copy import copy, deepcopy
from itertools import product

import numpy as np
import seaborn as sns
from nautilus import Prior, Sampler

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import photo_cov as pcov
from cosmicfishpie.LSSsurvey import photo_obs as pobs
from cosmicfishpie.utilities.utils import printing as upr

snscolors = sns.color_palette("colorblind")


def is_indexable_iterable(var):
    return isinstance(var, (list, np.ndarray, Sequence)) and not isinstance(var, (str, bytes))


logger = logging.getLogger("cosmicfishpie.cosmology.nuisance")
logger.setLevel(logging.INFO)


upr.debug = False


upr.debug_print("test")


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
    "outroot": "photo_3x2pt",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
    # "specs_dir": "../survey_specifications",
    "cosmo_model": "LCDM",
    "bfs8terms": False,
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


def observable_Cell(photo_th: pobs.ComputeCls):
    photo_th.compute_all()
    binrange = photo_th.binrange
    nbin = len(binrange)

    ells = photo_th.result["ells"]
    output = dict(ells=ells)

    observables = photo_th.observables
    if "WL" in observables:
        Cell_LL = np.empty((len(ells), nbin, nbin), dtype=np.float64)
    if "GCph" in observables:
        Cell_GG = np.empty((len(ells), nbin, nbin), dtype=np.float64)
    if "WL" in observables and "GCph" in observables:
        Cell_GL = np.empty((len(ells), nbin, nbin), dtype=np.float64)

    for i, j in product(binrange, repeat=2):

        if "WL" in observables:
            Cell_LL[:, i - 1, j - 1] = (
                photo_th.result["WL {}xWL {}".format(i, j)]
                + np.eye(nbin)[i - 1, j - 1]
                * photo_cov_fid.ellipt_error**2.0
                / photo_cov_fid.ngalbin[i - 1]
            )

        if "GCph" in observables:
            Cell_GG[:, i - 1, j - 1] = (
                photo_th.result["GCph {}xGCph {}".format(i, j)]
                + np.eye(nbin)[i - 1, j - 1] * 1 / photo_cov_fid.ngalbin[i - 1]
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


# In[12]:


Cells_fid = observable_Cell(photo_fid)


# In[13]:


ellmax_WL = cosmoFM_fid.specs["lmax_WL"]
ellmax_GC = cosmoFM_fid.specs["lmax_GCph"]
ellmax_XC = np.minimum(ellmax_GC, ellmax_WL)


# In[14]:


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


# In[15]:


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


# In[ ]:


samp1dic = {
    "Omegam": 0.31,
    "Omegab": 0.05,
    "h": 0.68,
    "ns": 0.96,
    "sigma8": 0.82,
    "w0": -1.01,
    "wa": 0.2,
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
    "AIA": 1.72,
    "betaIA": 2.17,
    "etaIA": -0.41,
}
print("Sample likelihood", loglike(samp1dic))


loglike(photo_cov_fid.allparsfid)


prior = Prior()
prior_withnuis = Prior()



prior_dict = {
    "Omegam": [0.24, 0.4],
    "Omegab": [0.04, 0.06],
    "h": [0.61, 0.73],
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


# In[ ]:


cosmoFM_fid.freeparams


# In[29]:


for par in prior_dict.keys():
    if par in cosmoFM_fid.freeparams.keys():
        dist_prior = (prior_dict[par][0], prior_dict[par][1])
        if re.match(r"b\d+", par):
            prior_withnuis.add_parameter(par, dist_prior)
        elif re.search(r"IA", par):
            prior_withnuis.add_parameter(par, dist_prior)
        else:
            prior.add_parameter(par, dist_prior)
            prior_withnuis.add_parameter(par, dist_prior)


# In[ ]:


prior.keys


# In[ ]:


print("Loading prior with keys: ", prior.keys)
sampler = Sampler(
    prior,
    loglike,
    n_live=1000,
    n_networks=4,
    n_batch=256,
    pool=24,
    pass_dict=False,
    filepath="cosmicshark_3x2photo_symb_justcosmopar.hdf5",
    resume=True,
    likelihood_kwargs={"prior": prior},
)
sampler.run(verbose=True, discard_exploration=True)
log_z_all = sampler.evidence()
print("Evidence:", log_z_all)
points_all, log_w_all, log_l_all = sampler.posterior()
sample_wghlkl = np.vstack((points_all.T, np.exp(log_w_all), log_l_all)).T
print("Printing chain to file.....")
np.savetxt("sample_wghlkl.txt", sample_wghlkl)


# In[ ]:
