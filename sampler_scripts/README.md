# Sampler examples

This source checkout retains one compact, tested symbolic LCDM input
(`fresh_lcdm_symbolic_small.yaml`) and the generic `run_sampler.py` runner as
reference examples for likelihood validation.

All operational sampler configurations, 3x2 drivers, and Slurm-facing test
assets live in `cosmicfishpie-validation/jobs_punch/sampler_scripts`. Run jobs
from that validation checkout so generated output is directed to JURECA scratch.
