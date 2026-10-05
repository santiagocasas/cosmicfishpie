#!/bin/bash
# Generic spawn-safe launcher for sampler_scripts/run_sampler.py.
#
# Submit from the repository root, for example:
#   sbatch --cpus-per-task=32 --export=ALL,REPO_ROOT=$PWD,\
#   CONFIG_FILE=sampler_scripts/fresh_wl_gcsp_symbolic_independent_32x1.yaml,\
#   OUTPUT_DIR=$SCRATCH/cfp/wl-gcsp-r001 scripts/slurm/nautilus_sampler.sbatch
#
# CFP_ENV_SETUP may name a site-local shell script that activates the project
# environment. The YAML pool must equal the Slurm CPU allocation.
# OUTPUT_DIR is stable across reruns so Nautilus can resume its HDF5 checkpoint.

#SBATCH --job-name=cfp-nautilus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --mem=32G

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the repository checkout}"
: "${CONFIG_FILE:?Set CONFIG_FILE to a YAML path relative to REPO_ROOT or absolute}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a persistent scratch/project directory}"

if [[ -n "${CFP_ENV_SETUP:-}" ]]; then
    source "$CFP_ENV_SETUP"
fi

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE"
fi

mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$REPO_ROOT"
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
srun --cpu-bind=cores python -u sampler_scripts/run_sampler.py \
    "$CONFIG_FILE" --output-dir "$OUTPUT_DIR"
