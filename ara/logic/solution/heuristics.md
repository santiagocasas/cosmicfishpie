# Heuristics


## H01: Start with the symbolic 32x1 smoke chain
- **Rationale**: `sample_symbolic_32x1.sbatch` completed job 15623397 in 7m50s wall time and produced a completed chain with ESS 10,003.77.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: medium
- **Code ref**: [`/p/home/jusers/casas1/jureca/cosmopunch/jobs_punch/sample_symbolic_32x1.sbatch`, `sampler_scripts/fresh_lcdm_symbolic_small.yaml`]
- **Proof**: [`evidence/tables/job_15623397_summary.csv`, `trace/exploration_tree.yaml:N02`]
- **From staging**: O01

## H02: Smooth coarse-binned contours before escalating posterior sampling
- **Rationale**: For this three-parameter Nautilus chain with N_eff about 2000, a 30-bin 2D histogram created visually bubbly contours. The project-style 12-bin, smooth=5 rendering resolved the visual artifact without altering samples.
- **Provenance**: user
- **Crystallized via**: empirical-resolution
- **Sensitivity**: medium
- **Code ref**: [cosmicfishpie/analysis/fishconsumer.py, chains/wl_gcsp_symbolic_independent_15631246/chains_fresh_wl_gcsp_symbolic_independent_32x1/wl_gcsp_independent_posterior_triangle_smooth12x5.png]
- **From staging**: O05

## H03: Match fixed-nuisance Nautilus contours to conditional Fishers
- **Rationale**: The suite samples only `Omegam`, `Omegab`, and `sigma8`; all unlisted nuisance parameters remain fixed. A conditional Fisher with the same Gaussian YAML prior on `Omegab` is therefore the direct posterior reference, while marginalizing nuisance parameters answers a different forecast question.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`sampler_scripts/run_3x2pt_fisher_nautilus_case.py`, `sampler_scripts/postprocess_3x2pt_fisher_nautilus.py`]
- **Proof**: [`trace/exploration_tree.yaml:N08`]
- **From staging**: O06
