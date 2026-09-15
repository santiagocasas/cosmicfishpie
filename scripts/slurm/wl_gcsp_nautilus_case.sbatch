#!/usr/bin/env bash
# Submit through submit_wl_gcsp_nautilus.sh; its sbatch options set resources.
# Required exported variables: REPO_ROOT, RUN_DIR, CASE, N_LIVE, N_EFF, WORKERS,
# SAMPLE_NUISANCES, NUISANCE_SIGMA, FREE_PARAMS.
set -euo pipefail

cd "$REPO_ROOT"

# Nautilus parallelises likelihood calls with multiprocessing. Prevent every
# worker from starting an additional BLAS/OpenMP team and oversubscribing CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
    scripts/run_wl_gcsp_nautilus_case.py
    --case "$CASE"
    --run-dir "$RUN_DIR"
    --free-params "$FREE_PARAMS"
    --n-live "$N_LIVE"
    --n-eff "$N_EFF"
    --workers "$WORKERS"
    --nuisance-sigma "$NUISANCE_SIGMA"
)
if [[ "$SAMPLE_NUISANCES" == "1" ]]; then
    args+=(--sample-nuisances)
fi

echo "Starting ${CASE}: ${SLURM_CPUS_PER_TASK:-unknown} CPUs allocated; ${WORKERS} Nautilus workers"
srun uv run python "${args[@]}"
