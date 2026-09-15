#!/usr/bin/env bash
# Submitted with an afterok dependency by submit_wl_gcsp_nautilus.sh.
set -euo pipefail

cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun uv run python scripts/postprocess_wl_gcsp_nautilus.py --run-dir "$RUN_DIR"
