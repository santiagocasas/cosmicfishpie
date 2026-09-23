#!/usr/bin/env bash
# Check the status of a WL/GCsp/joint/postprocess Nautilus job group submitted
# by submit_wl_gcsp_nautilus.sh, and print the chain-vs-Fisher comparison
# statistics once postprocessing has completed.
#
# This script is READ-ONLY (sacct + reading result files) - it never submits,
# cancels, or waits on anything. Run it yourself, as many times as you like.
#
# Usage:
#   scripts/slurm/check_wl_gcsp_status.sh <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>
#   scripts/slurm/check_wl_gcsp_status.sh --watch [SECONDS] <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>
#
# submit_wl_gcsp_nautilus.sh prints a ready-to-copy invocation of this script
# with the real job IDs after every submission.
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>
  $0 --watch [SECONDS] <RUN_DIR> <WL_JOBID> <GCSP_JOBID> <JOINT_JOBID> <POST_JOBID>

Modes:
  default             Print one status snapshot and exit.
  --watch [SECONDS]  Refresh the snapshot repeatedly (default: 30 seconds).
                      The loop stops when all jobs complete or one fails.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

WATCH=0
INTERVAL=30
if [[ "${1:-}" == "--watch" ]]; then
    WATCH=1
    shift
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        INTERVAL="$1"
        shift
    fi
fi

if [ "$#" -ne 5 ]; then
    usage >&2
    exit 1
fi

if [ "$WATCH" -eq 1 ]; then
    while true; do
        clear 2>/dev/null || true
        printf 'Live status — %s (refresh every %ss; press Ctrl-C to stop)\n\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "$INTERVAL"
        status=0
        output="$($0 "$@" 2>&1)" || status=$?
        printf '%s\n' "$output"

        if grep -q "All four jobs COMPLETED" <<<"$output"; then
            exit 0
        fi
        if grep -q "One or more jobs did not complete successfully" <<<"$output"; then
            exit 1
        fi
        sleep "$INTERVAL"
    done
fi

RUN_DIR="$1"
WL_JOB="$2"
GCSP_JOB="$3"
JOINT_JOB="$4"
POST_JOB="$5"
IDS="${WL_JOB},${GCSP_JOB},${JOINT_JOB},${POST_JOB}"

echo "=== sacct -j ${IDS} ==="
sacct -j "$IDS" --format=JobID,JobName%18,State,ExitCode,Elapsed
echo

# -X: parent job rows only (skip .batch/.0 substeps) for a clean state check.
declare -A STATE
while IFS='|' read -r jobid state; do
    STATE["$jobid"]="$state"
done < <(sacct -j "$IDS" -X --noheader --format=JobID,State --parsable2)

all_completed=1
any_bad=0
for label_id in "WL:$WL_JOB" "GCsp:$GCSP_JOB" "WL+GCsp:$JOINT_JOB" "postprocess:$POST_JOB"; do
    label="${label_id%%:*}"
    jobid="${label_id##*:}"
    state="${STATE[$jobid]:-UNKNOWN}"
    case "$state" in
        COMPLETED) ;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            any_bad=1
            all_completed=0
            echo "  [$label] job $jobid: $state - check $RUN_DIR/logs/*-${jobid}.err"
            ;;
        *)
            all_completed=0
            echo "  [$label] job $jobid: $state (not finished yet)"
            ;;
    esac
done

if [ "$any_bad" -eq 1 ]; then
    echo
    echo "One or more jobs did not complete successfully - see logs above before re-running."
    exit 1
fi

if [ "$all_completed" -ne 1 ]; then
    echo
    echo "Still in progress - re-run this script again in a bit."
    exit 0
fi

echo "All four jobs COMPLETED."
echo

STATS_FILE="$RUN_DIR/chain_statistics.json"
if [ ! -f "$STATS_FILE" ]; then
    echo "chain_statistics.json not found yet at: $STATS_FILE"
    echo "(postprocess job reported COMPLETED but the file is missing - check its log)"
    exit 1
fi

echo "=== Chain vs. Fisher comparison: $STATS_FILE ==="
python3 - "$STATS_FILE" << 'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)

params = payload["parameters"]
reference_kind = payload["reference_kind"]
print(f"Fisher reference: {reference_kind}")

for label, comp in payload["probes"].items():
    ess = comp["effective_sample_size"]
    precision = 100 * comp["sigma_fractional_error"]
    print(f"\n{label}  (ESS {ess:.0f}, sigma precision +-{precision:.0f}%)")
    print(f"  {'param':<10} {'mean':>12} {'sigma':>12} {'Fisher':>12} {'ratio':>8} {'bias':>8}")
    for i, name in enumerate(params):
        print(
            f"  {name:<10} {comp['mean'][i]:>12.6f} {comp['sigma'][i]:>12.6f} "
            f"{comp['sigma_fisher'][i]:>12.6f} {comp['sigma_ratio'][i]:>8.3f} "
            f"{comp['bias'][i]:>+8.2f}"
        )
    if len(params) == 2:
        print(
            f"  correlation: {comp['correlation'][0][1]:+.4f} (chain) vs "
            f"{comp['correlation_fisher'][0][1]:+.4f} (Fisher)"
        )

plot_dir = sys.argv[1].rsplit("/", 1)[0]
print("\nPlots:")
print(f"  {plot_dir}/nautilus_wl_gcsp_combined.png")
print(f"  {plot_dir}/fisher_vs_nautilus_wl_gcsp.png")
PYEOF
