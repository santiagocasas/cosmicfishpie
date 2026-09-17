import json
import multiprocessing
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Protocol

import numpy as np
from nautilus import Prior, Sampler
from scipy.stats import norm

from cosmicfishpie.configs.context import build_analysis_context
from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
from cosmicfishpie.likelihood.base import CompositeLikelihood, Likelihood
from cosmicfishpie.likelihood.photo_like import PhotometricLikelihood
from cosmicfishpie.likelihood.spectro_like import SpectroLikelihood

_WORKER_LIKELIHOOD = None


class LikelihoodComponent(Protocol):
    """Factory hooks needed to move likelihood data across spawn boundaries."""

    def build(self, config, settings, data=None, context=None) -> Likelihood:
        """Construct a likelihood around process-local cosmology state."""

    def data_payload(self, likelihood: Likelihood):
        """Return the serializable fiducial data needed by a spawned worker."""


class _PhotometricComponent:
    def build(self, config, settings, data=None, context=None):
        observables = settings.get("observables")
        if observables is None:
            observables = [obs for obs in config["observables"] if obs in ("WL", "GCph")]
        if context is None or set(context.observables) != set(observables):
            context = _build_context(config, observables=observables)
        return PhotometricLikelihood(
            cosmo_data=context,
            cosmo_theory=context,
            observables=observables,
            data_cells=data,
        )

    def data_payload(self, likelihood):
        return likelihood.data_obs


class _SpectroscopicComponent:
    def build(self, config, settings, data=None, context=None):
        observables = settings.get("observables")
        if observables is None:
            observables = [obs for obs in config["observables"] if obs in ("GCsp", "IM")]
        context = _build_fisher_context(config, observables)
        return SpectroLikelihood(
            cosmoFM_data=context,
            cosmoFM_theory=context,
            leg_flag=settings.get("leg_flag", "wedges"),
            data_obs=data,
            nuisance_shot=settings.get("nuisance_shot"),
            covariance_from=settings.get("covariance_from", "theory"),
        )

    def data_payload(self, likelihood):
        return likelihood.data_obs


LIKELIHOOD_COMPONENTS = {
    "photometric": _PhotometricComponent(),
    "spectroscopic": _SpectroscopicComponent(),
}


def register_likelihood_component(name: str, component: LikelihoodComponent) -> None:
    """Register a likelihood factory for use by serial and spawned samplers."""
    if not name or not isinstance(name, str):
        raise ValueError("Likelihood component names must be non-empty strings")
    if not callable(getattr(component, "build", None)) or not callable(
        getattr(component, "data_payload", None)
    ):
        raise TypeError("Likelihood components must define build() and data_payload()")
    LIKELIHOOD_COMPONENTS[name] = component


def _likelihood_specs(config):
    configured = config.get("likelihoods")
    if configured is None:
        observables = set(config["observables"])
        has_photo = bool(observables.intersection(("WL", "GCph")))
        has_spectro = bool(observables.intersection(("GCsp", "IM")))
        if has_photo and has_spectro:
            raise ValueError(
                "Mixed photometric and spectroscopic probes require an explicit "
                "'likelihoods' list; listing both declares them statistically independent"
            )
        if has_photo:
            return [{"type": "photometric", "settings": {}}]
        if has_spectro:
            return [{"type": "spectroscopic", "settings": {}}]
        raise ValueError("Could not infer a likelihood from the configured observables")

    if isinstance(configured, (str, dict)):
        configured = [configured]
    if not configured:
        raise ValueError("The 'likelihoods' list cannot be empty")

    specs = []
    for entry in configured:
        if isinstance(entry, str):
            spec = {"type": entry}
        elif isinstance(entry, dict):
            spec = dict(entry)
        else:
            raise TypeError("Each likelihood entry must be a component name or mapping")
        name = spec.pop("type", None)
        if name not in LIKELIHOOD_COMPONENTS:
            available = ", ".join(sorted(LIKELIHOOD_COMPONENTS))
            raise ValueError(f"Unknown likelihood component '{name}'. Available: {available}")
        specs.append({"type": name, "settings": spec})
    return specs


def _build_likelihood(specs, config, payloads=None, context=None):
    if payloads is None:
        payloads = [None] * len(specs)
    if len(payloads) != len(specs):
        raise ValueError("Likelihood component and payload counts do not match")

    likelihoods = []
    for spec, payload in zip(specs, payloads):
        component = LIKELIHOOD_COMPONENTS[spec["type"]]
        shared_context = context if len(specs) == 1 else None
        likelihoods.append(
            component.build(config, spec["settings"], payload, context=shared_context)
        )
    if len(likelihoods) == 1:
        return likelihoods[0]
    return CompositeLikelihood(likelihoods)


def _likelihood_payloads(specs, likelihood):
    likelihoods = (
        likelihood.likelihoods if isinstance(likelihood, CompositeLikelihood) else (likelihood,)
    )
    return [
        LIKELIHOOD_COMPONENTS[spec["type"]].data_payload(component_likelihood)
        for spec, component_likelihood in zip(specs, likelihoods)
    ]


def _build_context(config, observables=None):
    """Build an analysis context from a plain sampler configuration."""
    options = config["options"]
    return build_analysis_context(
        fiducialpars=config["fiducial"],
        options=options,
        observables=observables or config["observables"],
        cosmo_model=options["cosmo_model"],
        survey_name=options["survey_name"],
    )


def _build_fisher_context(config, observables):
    """Build mutable runtime state required by spectroscopic likelihoods."""
    options = config["options"]
    return FisherMatrix(
        options=deepcopy(options),
        observables=list(observables),
        fiducialpars=deepcopy(config["fiducial"]),
        surveyName=options["survey_name"],
        cosmoModel=options["cosmo_model"],
    )


def _initialize_likelihood_worker(config, likelihood_specs, data_payloads):
    """Construct process-local cosmology state in a freshly spawned worker."""
    global _WORKER_LIKELIHOOD
    _WORKER_LIKELIHOOD = _build_likelihood(likelihood_specs, config, data_payloads)


def _worker_loglike(param_dict):
    """Evaluate a sample using the likelihood initialized in this worker."""
    if _WORKER_LIKELIHOOD is None:
        raise RuntimeError("Likelihood worker was not initialized")
    return _WORKER_LIKELIHOOD.loglike(param_dict=param_dict)


def _format_param_label(param_name):
    if param_name.startswith("Omega"):
        suffix = param_name[len("Omega") :]
        return rf"$\Omega_{{{suffix}}}$" if suffix else r"$\Omega$"
    if param_name.startswith("omega"):
        suffix = param_name[len("omega") :]
        return rf"$\omega_{{{suffix}}}$" if suffix else r"$\omega$"
    if param_name.startswith("b") and param_name[1:].isdigit():
        return rf"$b_{{{param_name[1:]}}}$"

    label_map = {
        "Omegam": r"$\Omega_m$",
        "Omegab": r"$\Omega_b$",
        "Omegac": r"$\Omega_c$",
        "Omegak": r"$\Omega_k$",
        "sigma8": r"$\sigma_8$",
        "ns": r"$n_{\rm s}$",
        "w0": r"$w_0$",
        "wa": r"$w_a$",
        "h": r"$h$",
        "A_s": r"$10^9 A_s$",
        "As": r"$10^9 A_s$",
        "H0": r"$H_0$",
        "mnu": r"$m_\nu$",
        "Neff": r"$N_{\rm eff}$",
        "AIA": r"$A_{\rm IA}$",
        "etaIA": r"$\eta_{\rm IA}$",
    }
    return label_map.get(param_name, param_name)


def load_chain_metadata(chain_folder, metadata_filename=None, label_overrides=None):
    """Load chain file path and sampled fiducial parameters from metadata.

    Args:
        chain_folder (str): Path to the chains directory containing metadata.
        metadata_filename (str, optional): Specific metadata json filename.
        label_overrides (dict, optional): Map of param name to LaTeX label.

    Returns:
        tuple[str, dict, dict, dict]: (chain_file_path, sampled_fiducial_params, metadata, param_labels)
    """
    if metadata_filename is None:
        candidates = [f for f in os.listdir(chain_folder) if f.endswith("_metadata.json")]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one metadata file in " f"{chain_folder}, found: {candidates}"
            )
        metadata_filename = candidates[0]

    metadata_path = os.path.join(chain_folder, metadata_filename)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    chain_file = metadata.get("chain_file")
    if chain_file is None:
        outroot = metadata.get("outroot path")
        chain_file = f"{outroot}.txt" if outroot else None
    if chain_file is None:
        raise ValueError(f"Chain file not found in metadata: {metadata_path}")
    if not os.path.isabs(chain_file):
        chain_file = os.path.join(chain_folder, os.path.basename(chain_file))

    sampled_fiducial = metadata.get("sampled_fiducial_params", {})
    label_overrides = label_overrides or {}
    param_labels = {
        name: label_overrides.get(name, _format_param_label(name))
        for name in sampled_fiducial.keys()
    }
    return chain_file, sampled_fiducial, metadata, param_labels


class NautilusSampler:
    def __init__(self, config):
        self.config = config
        self.fiducial = config["fiducial"]
        self.observables = config["observables"]
        self.options = config["options"]
        self.prior_dict = config["priors"]
        self.sampler_settings = config["sampler_settings"]
        self.use_nuisance = config.get("use_nuisance", True)
        self.tini = None
        self.tfin = None

        # Setup output path
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        output_dir = self.config.get("output_dir", "chains")
        self.folder_name = os.path.join(output_dir, f"chains_{self.config['name']}")
        os.makedirs(self.folder_name, exist_ok=True)

        self.outroot = f"{self.folder_name}/cosmicjellyfish_{self.options['code']}_{self._get_survey_name()}_{self.config['name']}"
        self.options["outroot"] = self.outroot
        print("\n" + "-" * 50)
        print("Intialized NautilusSampler")
        print(f"📁 Output folder: {self.folder_name}")
        print(f"📄 Outroot path: {self.outroot}")
        print(f"⚙️ Pool threads: {self.sampler_settings['pool']}")
        print(f"📌 Config name: {self.config['name']}")
        print("-" * 50 + "\n")
        # Setup
        self._setup_cosmology()
        self._setup_priors()
        self._setup_likelihood()

    def _get_survey_name(self):
        return self.options["survey_name_photo"] or self.options["survey_name_spectro"]

    def _setup_cosmology(self):
        self.cosmo_context = _build_context(self.config)

    def _setup_priors(self):
        self.prior_chosen = Prior()
        for par, prior_range in self.prior_dict.items():
            if par in self.cosmo_context.freeparams:
                if isinstance(prior_range, dict) and prior_range["type"] == "gaussian":
                    dist = norm(loc=prior_range["loc"], scale=prior_range["scale"])
                else:
                    dist = tuple(prior_range)
                self.prior_chosen.add_parameter(par, dist)

    def _setup_likelihood(self):
        self.likelihood_specs = _likelihood_specs(self.config)
        self.likelihood = _build_likelihood(
            self.likelihood_specs, self.config, context=self.cosmo_context
        )
        self.likelihood_payloads = _likelihood_payloads(self.likelihood_specs, self.likelihood)

    def _save_metadata(self, evidence=None, finish_time=None):

        sampled_fiducial = {}
        for param in self.prior_dict.keys():
            if param in self.cosmo_context.allparams:
                sampled_fiducial[param] = self.cosmo_context.allparams[param]

        metadata = {
            "cosmicfishpie_version": "1.0.0",  # Replace with actual version
            "name": self.config["name"],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.tini)),
            "outroot path": self.outroot,
            "cosmo_fiducial_params": self.fiducial,
            "sampled_fiducial_params": sampled_fiducial,
            "priors": {k: str(v) for k, v in self.prior_dict.items()},
            "sampler_settings": self.sampler_settings,
            "observables": self.observables,
            "cosmo_model": self.options["cosmo_model"],
            "survey_name": self._get_survey_name(),
            "code": self.options["code"],
        }

        if finish_time:
            metadata.update(
                {
                    "finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_time)),
                    "elapsed_time": self._format_time(finish_time - self.tini),
                    "evidence_log_z": float(evidence) if evidence else None,
                    "chain_file": self.chain_file,
                }
            )

        with open(self.outroot + "_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def run(self):
        self.chain_file = self.outroot + ".txt"
        self.chain_hdf5 = self.outroot + ".hdf5"
        self.tini = time.time()
        self._save_metadata()
        worker_pool = None

        def stop_worker_pool(terminate=False):
            nonlocal worker_pool
            if worker_pool is None:
                return
            if terminate:
                worker_pool.terminate()
            else:
                worker_pool.close()
            worker_pool.join()
            worker_pool = None

        def start_sampler(parallel=True):
            nonlocal worker_pool
            pool_size = int(self.sampler_settings["pool"])
            sampler_kwargs = {
                "prior": self.prior_chosen,
                "n_live": self.sampler_settings["n_live"],
                "n_networks": self.sampler_settings["n_networks"],
                "n_batch": self.sampler_settings["n_batch"],
                "filepath": self.chain_hdf5,
                "resume": True,
            }

            if parallel and pool_size > 1:
                # Build heavy CAMB/CLASS state independently in fresh processes.
                # Only plain component specs and serializable fiducial data cross
                # the spawn boundary; the frozen AnalysisContext never does.
                worker_pool = multiprocessing.get_context("spawn").Pool(
                    pool_size,
                    initializer=_initialize_likelihood_worker,
                    initargs=(
                        self.config,
                        self.likelihood_specs,
                        self.likelihood_payloads,
                    ),
                )
                sampler_kwargs.update(
                    likelihood=_worker_loglike,
                    pool=worker_pool,
                    pass_dict=True,
                )
            else:
                sampler_kwargs.update(
                    likelihood=self.likelihood.loglike,
                    pool=None,
                    pass_dict=False,
                    likelihood_kwargs={"prior": self.prior_chosen},
                )

            try:
                return Sampler(**sampler_kwargs)
            except Exception:
                stop_worker_pool(terminate=True)
                raise

        if os.path.exists(self.chain_file):
            print(f"Chain file exists: {self.chain_file}")
            print("Run already completed. Updating metadata only!")
            if os.path.exists(self.chain_hdf5):
                try:
                    print("Loading sampler to get evidence...")
                    nautilus_sampler = start_sampler(parallel=False)
                    evidence = nautilus_sampler.evidence()
                    print(f"Evidence: {evidence:.2f}")
                    self._save_metadata(evidence=evidence, finish_time=time.time())
                    print("Metadata.json updated")
                except Exception as e:
                    print(f"WARNING: Could not load existing run: {e}")
            else:
                self._save_metadata(evidence=None, finish_time=time.time())
                print("Metadata.json updated")
            return

        print("Starting new run")
        try:
            nautilus_sampler = start_sampler()
            print("\n" + "=" * 60)
            print("🚀 Starting Nautilus Sampler Run")
            print("=" * 60)
            print(f"📁 Output folder: {self.folder_name}")
            print(f"📄 Outroot path: {self.outroot}")
            print(
                f"⏱️ Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"
            )
            print(f"⚙️ Pool threads: {self.sampler_settings['pool']}")
            print(f"🔬 Code: {self.options['code']}")
            print(f"📦 Config name: {self.config['name']}")
            print("=" * 60 + "\n")
            run_kwargs = {
                "verbose": self.sampler_settings.get("verbose", True),
                "discard_exploration": self.sampler_settings.get("discard_exploration", True),
            }
            for key in ("f_live", "n_shell", "n_eff", "n_like_max", "timeout"):
                if key in self.sampler_settings:
                    run_kwargs[key] = self.sampler_settings[key]
            completed = nautilus_sampler.run(**run_kwargs)
            if not completed:
                raise RuntimeError(
                    "Nautilus stopped before convergence (e.g. due to n_like_max or "
                    "timeout); resumable checkpoint was retained in "
                    f"{self.chain_hdf5}. Re-run to resume and reach convergence "
                    "before the final chain and metadata are saved."
                )
            evidence = nautilus_sampler.log_z
            points, log_w, log_l = nautilus_sampler.posterior()

            self.tfin = time.time()
            sample_wghlkl = np.vstack((points.T, np.exp(log_w), log_l)).T
            outfile_chain = self.outroot + ".txt"
            header = " ".join(self.prior_chosen.keys) + " weights loglike"
            np.savetxt(outfile_chain, sample_wghlkl, header=header)
            print(f"Saved chain to {outfile_chain}")
            self._save_metadata(evidence=evidence, finish_time=self.tfin)

        except Exception as e:
            stop_worker_pool(terminate=True)
            print(f"Sampler error: {e}")
            raise
        else:
            stop_worker_pool()
