#!/usr/bin/env bash
# Submit independent WL, GCsp, and joint Nautilus jobs plus dependent plotting.
# Usage example:
#   RUN_DIR=/scratch/$USER/cfp/wl-gcsp-500 N_LIVE=500 N_EFF=2000 \
#   CPUS_PER_TASK=16 WALLTIME=24:00:00 scripts/slurm/submit_wl_gcsp_nautilus.sh
#
# Set SAMPLE_NUISANCES=0 for the cheaper conditional-debugging run. The default
# is 1, which samples all Fisher nuisances and compares to marginal Fishers.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to an absolute scratch/project output directory}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
N_LIVE="${N_LIVE:-500}"
N_EFF="${N_EFF:-2000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
WALLTIME="${WALLTIME:-24:00:00}"
MEMORY="${MEMORY:-32G}"
SAMPLE_NUISANCES="${SAMPLE_NUISANCES:-1}"
NUISANCE_SIGMA="${NUISANCE_SIGMA:-5.0}"
FREE_PARAMS="${FREE_PARAMS:-Omegam,sigma8}"

mkdir -p "$RUN_DIR/logs"

common=(
    --parsable
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEMORY"
    --time="$WALLTIME"
    --output="$RUN_DIR/logs/%x-%j.out"
    --error="$RUN_DIR/logs/%x-%j.err"
)
exports="ALL,REPO_ROOT=$REPO_ROOT,RUN_DIR=$RUN_DIR,N_LIVE=$N_LIVE,N_EFF=$N_EFF"
exports+=",WORKERS=$CPUS_PER_TASK,SAMPLE_NUISANCES=$SAMPLE_NUISANCES"
exports+=",NUISANCE_SIGMA=$NUISANCE_SIGMA,FREE_PARAMS=$FREE_PARAMS"

submit_case() {
    local case="$1"
    sbatch "${common[@]}" --job-name="cfp-${case}" --export="$exports,CASE=$case" \
        "$REPO_ROOT/scripts/slurm/wl_gcsp_nautilus_case.sbatch"
}

wl_job="$(submit_case wl)"
gcsp_job="$(submit_case gcsp)"
joint_job="$(submit_case joint)"
dependency="afterok:${wl_job}:${gcsp_job}:${joint_job}"
post_job="$(sbatch --parsable --ntasks=1 --cpus-per-task=1 --mem=8G --time=01:00:00 \
    --job-name=cfp-postprocess --dependency="$dependency" \
    --output="$RUN_DIR/logs/%x-%j.out" --error="$RUN_DIR/logs/%x-%j.err" \
    --export="ALL,REPO_ROOT=$REPO_ROOT,RUN_DIR=$RUN_DIR" \
    "$REPO_ROOT/scripts/slurm/wl_gcsp_nautilus_postprocess.sbatch")"

cat <<EOF
Submitted:
  WL:          $wl_job
  GCsp:        $gcsp_job
  WL+GCsp:     $joint_job
  postprocess: $post_job (afterok on all three)

Run directory: $RUN_DIR
Logs:          $RUN_DIR/logs/
EOF
