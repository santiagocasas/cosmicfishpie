import os
import json
import time
from datetime import datetime
import numpy as np

from cosmicfishpie.fishermatrix import cosmicfish
from cosmicfishpie.LSSsurvey import photo_obs as pobs
from cosmicfishpie.LSSsurvey import photo_cov as pcov
from cosmicfishpie.likelihood import PhotometricLikelihood
from nautilus import Prior, Sampler
from scipy.stats import norm


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
        self.folder_name = f"chains/chains_{self.config['name']}"
        os.makedirs(self.folder_name, exist_ok=True)

        self.outroot = f"{self.folder_name}/cosmicjellyfish_{self.options['code']}_{self._get_survey_name()}_{self.config['name']}"
        self.options["outroot"] = self.outroot
        print("\n" + "-"*50)
        print("Intialized NautilusSampler")
        print(f"📁 Output folder: {self.folder_name}")
        print(f"📄 Outroot path: {self.outroot}")
        print(f"⚙️ Pool threads: {self.sampler_settings['pool']}")
        print(f"📌 Config name: {self.config['name']}")
        print("-"*50 + "\n")
        # Setup
        self._setup_cosmology()
        self._setup_priors()
        self._setup_likelihood()

    def _get_survey_name(self):
        return self.options["survey_name_photo"] or self.options["survey_name_spectro"]

    def _setup_cosmology(self):
        self.cosmoFM_fid = cosmicfish.FisherMatrix(
            fiducialpars=self.fiducial,
            options=self.options,
            observables=self.observables,
            cosmoModel=self.options["cosmo_model"],
            surveyName=self.options["survey_name"],
        )

    def _setup_priors(self):
        self.prior_chosen = Prior()
        for par, prior_range in self.prior_dict.items():
            if par in self.cosmoFM_fid.freeparams:
                if isinstance(prior_range, dict) and prior_range["type"] == "gaussian":
                    dist = norm(loc=prior_range["loc"], scale=prior_range["scale"])
                else:
                    dist = tuple(prior_range)
                self.prior_chosen.add_parameter(par, dist)

    def _setup_likelihood(self):
        self.photo_like = PhotometricLikelihood(cosmo_data=self.cosmoFM_fid, cosmo_theory=self.cosmoFM_fid)

    def _save_metadata(self, evidence=None, finish_time=None):

        sampled_fiducial = {}
        for param in self.prior_dict.keys():
            if param in self.cosmoFM_fid.allparams:
                sampled_fiducial[param] = self.cosmoFM_fid.allparams[param]

        metadata = {
            "cosmicfishpie_version": "1.0.0",  # Replace with actual version
            "name": self.config["name"],
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.tini)),
            "outroot path": self.outroot,
            "cosmo_fiducial_params": self.fiducial,
            "sampled_fiducial_params": sampled_fiducial,
            "priors": {k: str(v) for k, v in self.prior_dict.items()},
            "sampler_settings": self.sampler_settings,
            "observables": self.observables,
            "cosmo_model": self.options["cosmo_model"],
            "survey_name": self._get_survey_name(),
            "code": self.options["code"]
        }

        if finish_time:
            metadata.update({
                "finish_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time)),
                "elapsed_time": self._format_time(finish_time - self.tini),
                "evidence_log_z": float(evidence) if evidence else None,
                "chain_file": self.chain_file
            })

        with open(self.outroot + "_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def run(self):
        # Check if chain file exists (run completed)
        self.chain_file = self.outroot + ".txt"
        self.chain_hdf5 = self.outroot + ".hdf5"
        # Save initial metadata
        self.tini = time.time()
        self._save_metadata()
        # Set up sampler with resume capability
        nautilus_sampler = None
        def start_sampler():
            naut = Sampler(
                prior=self.prior_chosen,
                likelihood=self.photo_like.loglike,
                n_live=self.sampler_settings["n_live"],
                n_networks=self.sampler_settings["n_networks"],
                n_batch=self.sampler_settings["n_batch"],
                pool=self.sampler_settings["pool"],
                pass_dict=False,
                filepath=self.chain_hdf5,
                resume=True,  # This handles checkpointing automatically
                likelihood_kwargs={'prior': self.prior_chosen}
            )
            return naut
        
        if os.path.exists(self.chain_file):
            print(f"Chain file exists: {self.chain_file}")
            print("Run already completed. Updating metadata only!")
            # Load existing evidence and timing info
            if os.path.exists(self.chain_hdf5): 
                try:
                    print("Loading sampler to get evidence...")
                    nautilus_sampler = start_sampler()
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
        else: 
            print("Starting new run")
            nautilus_sampler = start_sampler()
        
        # Run sampler
        try:
            print("\n" + "="*60)
            print("🚀 Starting Nautilus Sampler Run")
            print("="*60)
            print(f"📁 Output folder: {self.folder_name}")
            print(f"📄 Outroot path: {self.outroot}")
            print(f"⏱️ Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
            print(f"⚙️ Pool threads: {self.sampler_settings['pool']}")
            print(f"🔬 Code: {self.options['code']}")
            print(f"📦 Config name: {self.config['name']}")
            print("="*60 + "\n")
            nautilus_sampler.run(verbose=True, discard_exploration=True)
            evidence = nautilus_sampler.log_z
            points, log_w, log_l = nautilus_sampler.posterior()
            
            self.tfin = time.time()
            # Save chain and final metadata
            sample_wghlkl = np.vstack((points.T, np.exp(log_w), log_l)).T
            outfile_chain = self.outroot + ".txt"
            header = "loglike weights " + " ".join(self.prior_chosen.keys)
            np.savetxt(outfile_chain, sample_wghlkl, header=header)
            print(f"Saved chain to {outfile_chain}")
            self._save_metadata(evidence=evidence, finish_time=self.tfin)
            
        except Exception as e:
            print(f"Sampler error: {e}")
            raise