# Sampler examples

This source checkout retains one compact, tested symbolic LCDM input
(`fresh_lcdm_symbolic_small.yaml`) and the generic `run_sampler.py` runner as
reference examples for likelihood validation.

All operational sampler configurations, 3x2 drivers, and Slurm-facing test
assets live in [cjfp-data/jobs_punch/sampler_scripts](https://dagshub.com/santiagocasas/cjfp-data/src/main/jobs_punch/sampler_scripts).
Follow that checkout's JURECA instructions for production jobs on scratch.

Run these source examples from the CosmicJellyfish repository root so the YAML
paths to survey specifications and solver settings resolve:

```bash
uv run python scripts/likelihood/sampler/run_sampler.py \
  scripts/likelihood/sampler/fresh_lcdm_symbolic_small.yaml
```

The generic sampler defaults to `./chains/chains_<name>/`; a YAML `output_dir`
or the runner's `--output-dir` overrides it. No chain samples are versioned here.
See the [data and workflow guide](../../../docs/source/validation_data.md).
