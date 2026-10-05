#!/bin/bash
#SBATCH --job-name=wlgcsp_postproc
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

cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# NOTE: plain python, not "uv run python" - uv is not installed on this
# cluster. common.sh already activated the correct .venv above. This step
# is single-threaded (just reads chains and makes plots); submit_wl_gcsp_
# nautilus.sh overrides --cpus-per-task=1 --mem=8G on the sbatch command
# line for this job, so the "32" above is only a fallback if run standalone.
srun python -u scripts/postprocess_wl_gcsp_nautilus.py --run-dir "$RUN_DIR"
