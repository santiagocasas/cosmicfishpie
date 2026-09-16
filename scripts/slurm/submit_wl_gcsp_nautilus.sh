#!/usr/bin/env bash
# Submit independent WL, GCsp, and joint Nautilus jobs plus dependent plotting.
# Usage example:
#   RUN_DIR=/scratch/$USER/cfp/wl-gcsp-500 N_LIVE=500 N_EFF=2000 \
#   CPUS_PER_TASK=16 WALLTIME=24:00:00 scripts/slurm/submit_wl_gcsp_nautilus.sh
#
# Set SAMPLE_NUISANCES=0 for the cheaper conditional-debugging run. The default
# is 1, which samples all Fisher nuisances and compares to marginal Fishers.
#
# NOTE: run_wl_gcsp_nautilus_case.py always uses the symbolic backend, which
# is embarrassingly parallel across Nautilus workers with no OpenMP benefit
# per call (unlike CLASS/CAMB). Historical scaling on this cluster shows
# symbolic throughput saturates around pool=32 - CPUS_PER_TASK beyond that
# mostly reserves idle cores on the (exclusive-node) dc-cpu partition rather
# than speeding anything up. Since dc-cpu allocates a whole physical node
# regardless of --cpus-per-task, there is no cost benefit to requesting less
# than 32 either. Pick CPUS_PER_TASK=32 unless you have a specific reason not to.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<HELPEOF
Usage: RUN_DIR=<dir> [OPTIONS] $0

Submits three independent Nautilus sampling jobs (WL, GCsp, joint) plus a
dependent postprocessing job comparing each chain with its Fisher reference.

Required:
  RUN_DIR            Absolute scratch/project output directory, for example:
                    RUN_DIR=\$SCRATCH_punch_astro/cfp/wl-gcsp-500

Optional environment variables:
  N_LIVE            Nautilus live points                  (default: 500)
  N_EFF             Effective-sample target                (default: 2000)
  CPUS_PER_TASK     Nautilus worker pool                   (default: 16;
                    prefer 32 for the symbolic backend)
  WALLTIME          Per-case time limit                    (default: 24:00:00)
  MEMORY            Per-case memory                        (default: 32G)
  SAMPLE_NUISANCES  1=sample nuisances, 0=fixed            (default: 1)
  NUISANCE_SIGMA    Gaussian nuisance-prior width           (default: 5.0)
  FREE_PARAMS       Comma-separated free parameters         (default: Omegam,sigma8)

Examples:
  # Cheap debug run:
  RUN_DIR=\$SCRATCH_punch_astro/cfp-debug/wl-gcsp-debug N_LIVE=100 N_EFF=400 \\
    CPUS_PER_TASK=32 SAMPLE_NUISANCES=0 $0

  # Full production run:
  RUN_DIR=\$SCRATCH_punch_astro/cfp/wl-gcsp-500 N_LIVE=500 N_EFF=2000 \\
    CPUS_PER_TASK=32 WALLTIME=24:00:00 $0

After submission, a ready-to-copy read-only status-check command is printed.
HELPEOF
    exit 0
fi

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
# NOTE: REPO_ROOT is exported below for parity/debugging only. The case and
# postprocess .sbatch scripts source jobs_punch/common.sh, which hardcodes
# its own REPO_ROOT and overwrites this exported value before the job does
# anything - so the actual working directory used by the job is always
# common.sh's path, not this one. This script's own REPO_ROOT (from git
# rev-parse) is only used below to locate the .sbatch files to submit.
exports="ALL,REPO_ROOT=$REPO_ROOT,RUN_DIR=$RUN_DIR,N_LIVE=$N_LIVE,N_EFF=$N_EFF"
exports+=",WORKERS=$CPUS_PER_TASK,SAMPLE_NUISANCES=$SAMPLE_NUISANCES"
# Slurm's --export has no quoting/escaping: it splits on every top-level
# comma to find VAR=VALUE pairs, so a raw comma-separated FREE_PARAMS value
# (e.g. "Omegam,sigma8") gets silently truncated to just "Omegam" once it
# crosses --export. Encode with ";" here; the case .sbatch decodes it back
# to commas before passing --free-params to the python CLI.
FREE_PARAMS_EXPORT="${FREE_PARAMS//,/;}"
exports+=",NUISANCE_SIGMA=$NUISANCE_SIGMA,FREE_PARAMS=$FREE_PARAMS_EXPORT"

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

cat <<INNEREOF
Submitted:
  WL:          $wl_job
  GCsp:        $gcsp_job
  WL+GCsp:     $joint_job
  postprocess: $post_job (afterok on all three)

Run directory: $RUN_DIR
Logs:          $RUN_DIR/logs/

Check status (safe to run anytime, read-only):
  $REPO_ROOT/scripts/slurm/check_wl_gcsp_status.sh "$RUN_DIR" $wl_job $gcsp_job $joint_job $post_job

Live status (refresh every 30 seconds; Ctrl-C to stop):
  $REPO_ROOT/scripts/slurm/check_wl_gcsp_status.sh --watch 30 "$RUN_DIR" $wl_job $gcsp_job $joint_job $post_job
INNEREOF
