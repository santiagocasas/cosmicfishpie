#!/bin/bash
#SBATCH --job-name=wlgc_nautilus
#SBATCH --account=punch_astro
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/p/home/jusers/casas1/jureca/cosmopunch/jobs_punch/logs/%x.%j.out
#SBATCH --error=/p/home/jusers/casas1/jureca/cosmopunch/jobs_punch/logs/%x.%j.err

set -euo pipefail
source /p/home/jusers/casas1/jureca/cosmopunch/jobs_punch/common.sh
prepare_cfp_environment
set_single_thread_numeric_libraries

# run_wl_gcsp_nautilus_case.py always uses the symbolic boltzmann backend
# (embarrassingly parallel across Nautilus workers, no OpenMP benefit per
# call - unlike CLASS/CAMB). Keep every worker single-threaded and let
# Nautilus multiprocessing use one core each; --cpu-bind=cores below pins
# each worker to its own physical core for cache locality.
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO_ROOT"

# The submit script encodes FREE_PARAMS with ";" instead of "," before
# putting it in --export (Slurm splits --export on every comma, so a raw
# comma-separated list would be silently truncated). Decode it back here.
FREE_PARAMS="${FREE_PARAMS//;/,}"

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
# NOTE: plain python, not "uv run python" - uv is not installed on this
# cluster (checked: not on PATH after full module load, no user-local
# install either). common.sh already activated the correct .venv above.
srun --cpu-bind=cores python -u "${args[@]}"
