#!/usr/bin/env bash
# Run the full validation matrix sequentially:
#   1) Casas et al. w0waCDM track (class vs camb, public CAMB 2.0.3)   -> env_01, env_02
#   2) Paper neutrino track (camb vs class, arXiv:2405.06047v1)        -> env_07..env_14
#
# Each case is invoked via scripts/compare_backends_report.sh --config <env file>.
# Per-case and total wall-clock timing is printed, and a pass/fail summary
# (based on each case's exit code, which reflects its SIGMA_THRESHOLD gate
# where set) is printed at the end. The script does NOT stop on a failing
# case - it runs every case and reports the full picture.
#
# After all cases finish, a machine-readable batch_summary.json is written and
# scripts/render_validation_dashboard.py refreshes the clean HTML dashboard
# covering the whole matrix, including per-parameter details and links to the
# exact YAML/common/Fisher specifications used by each case.
#
# Usage:
#   bash scripts/archive/run_all_validations.sh
#   OMP_NUM_THREADS=4 bash scripts/archive/run_all_validations.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/scripts/validation_configs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

BATCH_ID="validation_batch_$(date -u +%Y%m%d_%H%M%S)"
BATCH_DIR="${REPO_ROOT}/scripts/benchmark_results/${BATCH_ID}"
mkdir -p "${BATCH_DIR}"

# label                                    config file
CASES=(
  "01_w0waCDM_class_vs_camb_photo|compare_run_config.env_01_class_camb_photo_mpvalidation_w0waCDM"
  "02_w0waCDM_class_vs_camb_spectro|compare_run_config.env_02_class_camb_spectro_mpvalidation_w0waCDM"
  "07_paper_LCDM_fixed_photo|compare_run_config.env_07_camb_class_photo_papervalidation_LCDM_fixed"
  "08_paper_LCDM_fixed_spectro|compare_run_config.env_08_camb_class_spectro_papervalidation_LCDM_fixed"
  "09_paper_LCDM_mnu_photo|compare_run_config.env_09_camb_class_photo_papervalidation_LCDM_mnu"
  "10_paper_LCDM_mnu_spectro|compare_run_config.env_10_camb_class_spectro_papervalidation_LCDM_mnu"
  "11_paper_LCDM_mnu_Neff_photo|compare_run_config.env_11_camb_class_photo_papervalidation_LCDM_mnu_Neff"
  "12_paper_LCDM_mnu_Neff_spectro|compare_run_config.env_12_camb_class_spectro_papervalidation_LCDM_mnu_Neff"
  "13_paper_w0wa_mnu_Neff_photo|compare_run_config.env_13_camb_class_photo_papervalidation_w0wa_mnu_Neff"
  "14_paper_w0wa_mnu_Neff_spectro|compare_run_config.env_14_camb_class_spectro_papervalidation_w0wa_mnu_Neff"
)

declare -a RESULT_LABELS=()
declare -a RESULT_CONFIGS=()
declare -a RESULT_STATUS=()
declare -a RESULT_EXIT_CODES=()
declare -a RESULT_SECONDS=()
declare -a RESULT_OUTDIRS=()
declare -a RESULT_LOGS=()

fmt_hms() {
  local total="$1"
  printf '%02dh:%02dm:%02ds' "$((total/3600))" "$(((total%3600)/60))" "$((total%60))"
}

overall_start=$(date +%s)
echo "========================================================================"
echo "Running ${#CASES[@]} validation cases (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================================"

for entry in "${CASES[@]}"; do
  label="${entry%%|*}"
  config_file="${entry#*|}"
  config_path="${CONFIG_DIR}/${config_file}"

  echo
  echo "------------------------------------------------------------------------"
  echo "[${label}] starting  ($(date -u '+%H:%M:%S UTC'))"
  echo "[${label}] config: ${config_path}"
  echo "------------------------------------------------------------------------"

  log_file="${BATCH_DIR}/${label}.log"

  case_start=$(date +%s)
  bash "${REPO_ROOT}/scripts/compare_backends_report.sh" --config "${config_path}" 2>&1 | tee "${log_file}"
  status="${PIPESTATUS[0]}"
  case_end=$(date +%s)
  case_elapsed=$((case_end - case_start))

  if [[ ${status} -eq 0 ]]; then
    status_str="PASS"
  else
    status_str="FAIL(exit=${status})"
  fi

  # Recover the exact OUTDIR compare_backends_report.sh used, from its own
  # stdout line ("[compare] Running backends into: <outdir>"), rather than
  # re-deriving the config hash ourselves.
  outdir="$(grep -m1 '^\[compare\] Running backends into: ' "${log_file}" | sed 's/^\[compare\] Running backends into: //')"

  RESULT_LABELS+=("${label}")
  RESULT_CONFIGS+=("${config_file}")
  RESULT_STATUS+=("${status_str}")
  RESULT_EXIT_CODES+=("${status}")
  RESULT_SECONDS+=("${case_elapsed}")
  RESULT_OUTDIRS+=("${outdir}")
  RESULT_LOGS+=("${log_file}")

  echo "[${label}] finished: ${status_str}  elapsed: $(fmt_hms "${case_elapsed}")"
  if [[ -n "${outdir}" ]]; then
    echo "[${label}] outdir: ${outdir}"
  else
    echo "[${label}] WARNING: could not recover outdir from log ${log_file}"
  fi

  running_total=$(( $(date +%s) - overall_start ))
  echo "[${label}] cumulative elapsed so far: $(fmt_hms "${running_total}")"
done

overall_end=$(date +%s)
overall_elapsed=$((overall_end - overall_start))
finished_ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo
echo "========================================================================"
echo "Validation matrix summary"
echo "========================================================================"
printf '%-40s %-16s %s\n' "CASE" "STATUS" "ELAPSED"
for i in "${!RESULT_LABELS[@]}"; do
  printf '%-40s %-16s %s\n' "${RESULT_LABELS[$i]}" "${RESULT_STATUS[$i]}" "$(fmt_hms "${RESULT_SECONDS[$i]}")"
done
echo "------------------------------------------------------------------------"
echo "Total wall-clock time: $(fmt_hms "${overall_elapsed}")"
echo "Finished: ${finished_ts}"

# Write a machine-readable batch summary (labels, configs, exit codes,
# elapsed times, and each case's exact output directory) so it can be
# cross-referenced or re-rendered later without re-running anything.
summary_json="${BATCH_DIR}/batch_summary.json"
python3 - "${summary_json}" "${overall_elapsed}" "${finished_ts}" "${OMP_NUM_THREADS}" \
  "${RESULT_LABELS[@]}" -- "${RESULT_CONFIGS[@]}" -- "${RESULT_STATUS[@]}" -- \
  "${RESULT_EXIT_CODES[@]}" -- "${RESULT_SECONDS[@]}" -- "${RESULT_OUTDIRS[@]}" -- \
  "${RESULT_LOGS[@]}" <<'PYEOF'
import json
import sys

argv = sys.argv[1:]
out_path, overall_elapsed, finished_ts, omp_threads = argv[0], argv[1], argv[2], argv[3]
rest = argv[4:]

def take_until_sep(items):
    if "--" not in items:
        return items, []
    idx = items.index("--")
    return items[:idx], items[idx + 1:]

labels, rest = take_until_sep(rest)
configs, rest = take_until_sep(rest)
statuses, rest = take_until_sep(rest)
exit_codes, rest = take_until_sep(rest)
seconds, rest = take_until_sep(rest)
outdirs, rest = take_until_sep(rest)
logs = rest

cases = []
for i in range(len(labels)):
    cases.append({
        "label": labels[i],
        "config": configs[i] if i < len(configs) else None,
        "status": statuses[i] if i < len(statuses) else None,
        "exit_code": int(exit_codes[i]) if i < len(exit_codes) and exit_codes[i] != "" else None,
        "elapsed_seconds": int(seconds[i]) if i < len(seconds) and seconds[i] != "" else None,
        "outdir": outdirs[i] if i < len(outdirs) and outdirs[i] else None,
        "log_file": logs[i] if i < len(logs) else None,
    })

payload = {
    "overall_elapsed_seconds": int(overall_elapsed),
    "finished": finished_ts,
    "omp_num_threads": omp_threads,
    "cases": cases,
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=False)
print(f"[batch] Wrote batch summary: {out_path}")
PYEOF

echo
echo "[batch] Refreshing validation dashboard..."
render_py="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  render_py="${REPO_ROOT}/.venv/bin/python"
fi
dashboard_failed=0
if "${render_py}" "${REPO_ROOT}/scripts/render_validation_dashboard.py"; then
  echo "[batch] Validation dashboard: ${REPO_ROOT}/scripts/benchmark_results/dashboard/index.html"
else
  echo "[batch] WARNING: validation dashboard generation failed." >&2
  dashboard_failed=1
fi

echo "[batch] Batch directory: ${BATCH_DIR}"

if [[ ${dashboard_failed} -ne 0 ]]; then
  exit 1
fi

# Nonzero overall exit if any case failed its threshold gate.
for s in "${RESULT_STATUS[@]}"; do
  if [[ "${s}" != "PASS" ]]; then
    exit 1
  fi
done
exit 0
