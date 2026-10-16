(likelihood-sampling)=
# Likelihood sampling with Nautilus

A Fisher forecast approximates the posterior as a Gaussian around the fiducial
point. The likelihood subsystem drops that approximation: it recomputes the
observables at every trial parameter vector and samples the full posterior with
the [Nautilus](https://nautilus-sampler.readthedocs.io/en/latest/) nested sampler.
Use it to check a Fisher forecast, to capture non-Gaussian posteriors, or to get
the Bayesian evidence. The data are the noise-free prediction at the fiducial
cosmology, so posteriors are centred on the fiducial values.

:::{admonition} Expect it to be slow
:class: warning

Sampling needs thousands of likelihood evaluations, each recomputing the theory.
The examples below are deliberately tiny and still take **5 minutes or more on
one core**. Realistic runs need many cores: in {doc}`likelihood_validation` a
5-parameter WL run took 17 minutes on 64 workers and the 13-parameter WL+GCsp run
1 h 45 min. Develop locally, then run production chains on a cluster (Slurm
launchers are listed in {doc}`scripts_reference`). For quick exploration use a
Fisher forecast ({doc}`first_forecast`).
:::

## Minimal example

Run from the repository root so that `specs_dir` resolves. `n_live` and `n_eff`
are far too small for a real posterior.

```python
from cosmicfishpie.fishermatrix import cosmicfish as cff
from cosmicfishpie.likelihood import PhotometricLikelihood

options = {
    "accuracy": 1, "feedback": 1, "code": "symbolic", "cosmo_model": "LCDM",
    "ell_sampling": 25, "nonlinear": True, "outroot": "likelihood_demo", "results_dir": "results",
    "specs_dir": "cosmicfishpie/configs/default_survey_specifications",
    "survey_name": "Euclid",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "survey_name_spectro": False,
}
fiducial = {"Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "sigma8": 0.815584}

# Only supplies the survey and cosmology setup; no Fisher matrix is computed.
fm = cff.FisherMatrix(
    fiducialpars=fiducial, freepars={"Omegam": 0.01, "sigma8": 0.01}, options=options,
    observables=["WL"], cosmoModel="LCDM", surveyName="Euclid",
)

like = PhotometricLikelihood(cosmo_data=fm, cosmo_theory=fm)
print(like.loglike(param_dict=fiducial))  # ~0 at the fiducial point

prior = like.create_nautilus_prior({"Omegam": (0.28, 0.36), "sigma8": (0.75, 0.87)})
sampler = like.run_nautilus(
    prior=prior,
    sampler_kwargs={"n_live": 50, "n_networks": 1, "n_batch": 50},
    run_kwargs={"n_eff": 100, "verbose": False, "discard_exploration": True},
)
points, log_w, log_l = sampler.posterior()  # weighted posterior samples
print(points.shape, sampler.log_z)
```

`loglike` returns $\ln\mathcal{L} = -\chi^2/2$; parameters missing from
`param_dict` stay at their fiducial value. Nautilus needs at least two sampled
parameters, and `run_nautilus` is serial. For galaxy clustering use
{py:class}`~cosmicfishpie.likelihood.spectro_like.SpectroLikelihood` with
`observables=["GCsp"]`; it takes `cosmoFM_data=` and `cosmoFM_theory=` instead of
`cosmo_data=` and `cosmo_theory=`.

## Config-driven runs

For production, describe the run in one dictionary (or a YAML file; see
`sampler_scripts/fresh_lcdm_symbolic_small.yaml` and `sampler_scripts/run_sampler.py`).
{py:class}`~cosmicfishpie.likelihood.sampler.NautilusSampler` builds the
likelihood and priors, runs Nautilus (with a worker pool if `pool > 1`) and
writes the chain. This configuration also takes about five minutes on one core.

```python
from cosmicfishpie.likelihood import NautilusSampler

config = {
    "name": "wl_demo",
    "fiducial": fiducial,      # as above
    "observables": ["WL"],
    "options": dict(options),  # as above
    "priors": {
        "Omegam": [0.28, 0.36],  # uniform range
        "sigma8": {"type": "gaussian", "loc": 0.8156, "scale": 0.03},
    },
    "sampler_settings": {"n_live": 50, "n_networks": 1, "n_batch": 50, "pool": 1,
                         "n_eff": 100, "verbose": False},
}

if __name__ == "__main__":  # required for pool > 1: workers are spawned
    NautilusSampler(config).run()
```

`n_live`, `n_networks`, `n_batch` and `pool` are required; set `pool` to the
cores you have. `f_live`, `n_shell`, `n_eff`, `n_like_max`, `timeout`, `verbose`
and `discard_exploration` are passed on to Nautilus when present. Output goes to
`chains/chains_<name>/` (change the parent with `output_dir`): the chain (`*.txt`),
a resumable checkpoint (`*.hdf5`) and `*_metadata.json`.

Mistakes to avoid:

- **Re-running with the same `name` does not resample.** If the chain `.txt`
  exists, `run()` only refreshes the metadata and says so. Change `name`, or set
  `"overwrite": True` in the config to delete the old chain and sample again.
- **Only the model's default free parameters can be sampled** (`LCDM`: `Omegam`,
  `Omegab`, `h`, `ns`, `sigma8`). A prior on any other name is ignored with a
  warning, and fewer than two sampled parameters raises a `ValueError`.
- If sampling stops early (`n_like_max`, `timeout`), `run()` raises but keeps the
  checkpoint; re-running resumes.

## Post-processing

The chain header lists the parameter names, then `weights loglike` (linear
weights). {py:func}`~cosmicfishpie.likelihood.sampler.load_chain_metadata` returns
the chain path, the sampled fiducial values, the metadata (log-evidence, run time)
and LaTeX labels; the folder must hold exactly one `*_metadata.json`, otherwise
pass `metadata_filename`.

```python
from cosmicfishpie.analysis.fishconsumer import load_Nautilus_chains_from_txt, make_triangle_plot
from cosmicfishpie.likelihood.sampler import load_chain_metadata

chain_file, sampled_fiducial, metadata, labels = load_chain_metadata("chains/chains_wl_demo")
print(metadata["evidence_log_z"], metadata["elapsed_time"])

columns = open(chain_file).readline().lstrip("# ").split()[:-2]  # parameter names
chain = load_Nautilus_chains_from_txt(chain_file, columns)  # DataFrame

make_triangle_plot(chains=[chain], chain_labels=["Nautilus"], params=columns,
                   truth_values=sampled_fiducial, param_labels=labels, savefile="triangle.png")
```

In the DataFrame the `posterior` column (name kept for ChainConsumer) holds the
log-likelihood. To
overlay a Fisher forecast, also pass `fishers=[...]` and `fisher_labels=[...]`;
a chain that samples only cosmology fixes the nuisance parameters, so compare it
with the *conditional* Fisher, not the marginal one. From the shell,
`scripts/likelihood/postprocess/plot_finished_chain.py <run>_metadata.json` plots
finished chains.

## Combining probes

Mixing photometric and spectroscopic probes needs an explicit `likelihoods` list,
because summing their log-likelihoods declares them statistically independent.
Each entry is a component name, or a mapping whose extra keys go to that
component; several entries are wrapped in a
{py:class}`~cosmicfishpie.likelihood.base.CompositeLikelihood`:

```python
config["observables"] = ["WL", "GCsp"]
config["likelihoods"] = ["photometric", {"type": "spectroscopic", "leg_flag": "wedges"}]
```

A hand-built `CompositeLikelihood` has no `run_nautilus`; use `NautilusSampler`.

## Custom components

A custom component is a class with `build(config, settings, data=None, context=None)`
(returns a {py:class}`~cosmicfishpie.likelihood.base.Likelihood`) and
`data_payload(likelihood)` (returns the serialisable fiducial data that workers get
back as `data`). Register it with
{py:func}`~cosmicfishpie.likelihood.sampler.register_likelihood_component` at
import time, not under `if __name__ == "__main__"`, so spawned workers see it, then
select it with `config["likelihoods"] = [{"type": "<name>", ...}]`.

## Where to go next

- {doc}`likelihood_validation`: Fisher versus Nautilus for WL, GCsp and WL+GCsp, with run times.
- {doc}`scripts_reference`: the demo, Slurm case runner and post-processing scripts.
- {doc}`api/index`: API reference for `cosmicfishpie.likelihood`.
