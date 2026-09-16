# Likelihood Validation Results

This directory stores the compact, versioned evidence for the Euclid symbolic
WL/GCsp likelihood validation run documented in
`docs/source/likelihood_validation.md`.

`wl_gcsp_symbolic_euclid_64cpu_statistics.json` contains weighted Nautilus
posterior moments, their Fisher comparison, and effective sample sizes. The
raw Nautilus chains and HDF5 checkpoints are deliberately excluded because they
are large, machine-specific run artifacts.

Regenerate this file and the documentation figures from a copied completed HPC
run with:

```bash
uv run python scripts/render_likelihood_validation.py \
  --run-dir hpc-results/wl-gcsp-550-64cpu
```
