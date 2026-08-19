#!/usr/bin/env bash
# Run selected CosmicFishPie backend validation cases sequentially.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/scripts/validation_configs"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_selected_validations.sh --cases LIST [OPTIONS]
  bash scripts/run_selected_validations.sh --all [OPTIONS]

Select cases with comma-separated numbers, for example:
  bash scripts/run_selected_validations.sh --cases 7,8,11,12
  bash scripts/run_selected_validations.sh --cases 7 --omp-threads 4
  bash scripts/run_selected_validations.sh --all

Cases:
  01  Casas/MP w0waCDM, photo, class vs CAMB, informational (no gate)
  02  Casas/MP w0waCDM, spectro, class vs CAMB, informational (no gate)
  07  Paper LCDM, fixed 60 meV one-species neutrino, photo, gate <5%
  08  Paper LCDM, fixed 60 meV one-species neutrino, spectro, gate <2%
  09  Paper LCDM with free mnu, photo, gate <5%
  10  Paper LCDM with free mnu, spectro, gate <2%
  11  Paper LCDM with free mnu and Neff, photo, gate <5%
  12  Paper LCDM with free mnu and Neff, spectro, gate <2%
  13  Paper w0waCDM with free mnu and Neff, photo, gate <5%
  14  Paper w0waCDM with free mnu and Neff, spectro, gate <2%

Options:
  --cases LIST          Cases to run, e.g. 7,8,11,12. May be repeated.
  --all                 Run every available case listed above.
  --omp-threads N       Set OMP_NUM_THREADS (default: existing value or 8).
  --help                Show this help text.

Each case runs through compare_backends_report.sh, writes its own backend
comparison output, and is logged under scripts/benchmark_results/.
The HTML dashboard under scripts/benchmark_results/dashboard/ is refreshed
after all selected cases finish.
The script continues after a failed case and exits nonzero if any selected
case fails or cannot be started.
EOF
}

declare -A CASE_CONFIGS=(
  [01]="compare_run_config.env_01_class_camb_photo_mpvalidation_w0waCDM"
  [02]="compare_run_config.env_02_class_camb_spectro_mpvalidation_w0waCDM"
  [07]="compare_run_config.env_07_camb_class_photo_papervalidation_LCDM_fixed"
  [08]="compare_run_config.env_08_camb_class_spectro_papervalidation_LCDM_fixed"
  [09]="compare_run_config.env_09_camb_class_photo_papervalidation_LCDM_mnu"
  [10]="compare_run_config.env_10_camb_class_spectro_papervalidation_LCDM_mnu"
  [11]="compare_run_config.env_11_camb_class_photo_papervalidation_LCDM_mnu_Neff"
  [12]="compare_run_config.env_12_camb_class_spectro_papervalidation_LCDM_mnu_Neff"
  [13]="compare_run_config.env_13_camb_class_photo_papervalidation_w0wa_mnu_Neff"
  [14]="compare_run_config.env_14_camb_class_spectro_papervalidation_w0wa_mnu_Neff"
)

declare -a SELECTED_CASES=()
omp_threads="${OMP_NUM_THREADS:-8}"
all_cases=false

append_cases() {
  local value case_number normalized
  IFS=',' read -r -a requested <<< "$1"
  for value in "${requested[@]}"; do
    normalized="$(printf '%02d' "$((10#$value))" 2>/dev/null)" || {
      echo "Invalid case number: ${value}" >&2
      return 2
    }
    case_number="${normalized#0}"
    [[ "${normalized}" == "00" ]] && case_number="0"
    if [[ -z "${CASE_CONFIGS[${normalized}]+x}" ]]; then
      echo "Unknown case: ${value}" >&2
      return 2
    fi
    SELECTED_CASES+=("${normalized}")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      [[ $# -ge 2 ]] || { echo "--cases requires a value" >&2; exit 2; }
      append_cases "$2" || exit 2
      shift 2
      ;;
    --cases=*)
      append_cases "${1#*=}" || exit 2
      shift
      ;;
    --all)
      all_cases=true
      shift
      ;;
    --omp-threads)
      [[ $# -ge 2 ]] || { echo "--omp-threads requires a value" >&2; exit 2; }
      omp_threads="$2"
      shift 2
      ;;
    --omp-threads=*)
      omp_threads="${1#*=}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${all_cases}" == true ]]; then
  SELECTED_CASES=(01 02 07 08 09 10 11 12 13 14)
fi

if [[ ${#SELECTED_CASES[@]} -eq 0 ]]; then
  echo "Select cases with --cases LIST or use --all." >&2
  usage >&2
  exit 2
fi

export OMP_NUM_THREADS="${omp_threads}"
BATCH_ID="selected_validation_$(date -u +%Y%m%d_%H%M%S)"
BATCH_DIR="${REPO_ROOT}/scripts/benchmark_results/${BATCH_ID}"
mkdir -p "${BATCH_DIR}"

overall_start=$(date +%s)
failed=0
echo "Running cases: ${SELECTED_CASES[*]}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "Batch directory: ${BATCH_DIR}"

for case_number in "${SELECTED_CASES[@]}"; do
  config_file="${CASE_CONFIGS[${case_number}]}"
  config_path="${CONFIG_DIR}/${config_file}"
  log_file="${BATCH_DIR}/case_${case_number}.log"
  case_start=$(date +%s)

  echo
  echo "------------------------------------------------------------------------"
  echo "Case ${case_number}: ${config_file}"
  echo "------------------------------------------------------------------------"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" | tee "${log_file}"
    status=2
  else
    bash "${REPO_ROOT}/scripts/compare_backends_report.sh" \
      --config "${config_path}" 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"
  fi

  elapsed=$(( $(date +%s) - case_start ))
  if [[ ${status} -eq 0 ]]; then
    echo "Case ${case_number}: PASS (${elapsed}s)"
  else
    echo "Case ${case_number}: FAIL (exit ${status}, ${elapsed}s)"
    failed=1
  fi
done

overall_elapsed=$(( $(date +%s) - overall_start ))
echo
echo "Selected validation run finished in ${overall_elapsed}s."
echo "Logs and batch artifacts: ${BATCH_DIR}"

render_py="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  render_py="${REPO_ROOT}/.venv/bin/python"
fi
if "${render_py}" "${REPO_ROOT}/scripts/render_validation_dashboard.py"; then
  echo "Validation dashboard: ${REPO_ROOT}/scripts/benchmark_results/dashboard/index.html"
else
  echo "WARNING: validation dashboard generation failed." >&2
  failed=1
fi

exit "${failed}"
