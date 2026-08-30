#!/usr/bin/env bash
# Run selected CosmicFishPie backend validation cases sequentially.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/scripts/validation_configs"

# Validation cases are discovered automatically from config files named
# compare_run_config.env_<ID>_<description> in CONFIG_DIR -- adding a new case
# only requires dropping a new config file there, no edits to this script.
# <ID> is a dotted hierarchical case number (e.g. 05.3, 03.2.1); the leading
# root segment is zero-padded to 2 digits.
declare -A CASE_CONFIGS=()
declare -a CASE_ORDER=()

discover_cases() {
  local path filename case_number
  CASE_CONFIGS=()
  CASE_ORDER=()
  for path in "${CONFIG_DIR}"/compare_run_config.env_*; do
    [[ -f "${path}" ]] || continue
    filename="$(basename "${path}")"
    if [[ "${filename}" =~ ^compare_run_config\.env_([0-9]{2,}(\.[0-9]+)*)_ ]]; then
      case_number="${BASH_REMATCH[1]}"
      if [[ -n "${CASE_CONFIGS[${case_number}]+x}" ]]; then
        echo "Duplicate case number ${case_number}: ${CASE_CONFIGS[${case_number}]} and ${filename}" >&2
        return 1
      fi
      CASE_CONFIGS["${case_number}"]="${filename}"
      CASE_ORDER+=("${case_number}")
    fi
  done
  if [[ ${#CASE_ORDER[@]} -gt 0 ]]; then
    mapfile -t CASE_ORDER < <(printf '%s\n' "${CASE_ORDER[@]}" | sort -V)
  fi
}

# One-line case summary for --help: the config's leading "#" comment plus its
# SIGMA_THRESHOLD gate, both read live from the file so this never goes stale.
case_description() {
  local config_path="$1" line threshold
  IFS= read -r line < "${config_path}" 2>/dev/null || true
  line="${line#\#}"
  line="${line# }"
  threshold="$(grep -m1 '^SIGMA_THRESHOLD=' "${config_path}" 2>/dev/null | sed -E 's/^SIGMA_THRESHOLD="?([0-9.]+)"?.*/\1/')"
  if [[ -n "${threshold}" ]]; then
    printf '%s (gate <%s%%)' "${line}" "${threshold}"
  else
    printf '%s' "${line}"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_selected_validations.sh --cases LIST [OPTIONS]
  bash scripts/run_selected_validations.sh --all [OPTIONS]

Select cases with comma-separated dotted case IDs, for example:
  bash scripts/run_selected_validations.sh --cases 03.1,03.2
  bash scripts/run_selected_validations.sh --cases 03.2 --omp-threads 4
  bash scripts/run_selected_validations.sh --all

A group prefix (e.g. 03) expands to every discovered case under it (03.1,
03.2, 03.2.1, ...). An exact case ID (e.g. 03.2) selects only that case, even
if it also has sub-cases of its own.

Cases (auto-discovered from scripts/validation_configs/compare_run_config.env_<ID>_*):
EOF
  local case_number
  for case_number in "${CASE_ORDER[@]}"; do
    printf '  %-6s %s\n' "${case_number}" "$(case_description "${CONFIG_DIR}/${CASE_CONFIGS[${case_number}]}")"
  done
  cat <<'EOF'

Options:
  --cases LIST          Cases/groups to run, e.g. 03.1,03.2 or 03. May be repeated.
  --all                 Run every discovered case listed above.
  --omp-threads N       Set OMP_NUM_THREADS (default: existing value or 8).
  --force               Rerun cases even when an unchanged completed result exists.
  --verbose             Stream detailed backend output; default output is concise.
  --help                Show this help text.

Each case runs through compare_backends_report.sh, writes its own backend
comparison output, and is logged under scripts/benchmark_results/.
The HTML dashboard under scripts/benchmark_results/dashboard/ is refreshed
after all selected cases finish.
By default, completed cases are reused when their numerical inputs, relevant
code, backend versions, and saved run configuration still match. Partial or
stale cases are run again.
The script continues after a failed case and exits nonzero if any selected
case fails or cannot be started.

To add a new validation case, drop a new
scripts/validation_configs/compare_run_config.env_<ID>_<description> file --
no changes to this script are required. <ID> is a dotted hierarchical case
number, e.g. 05.3 or 03.2.1; the leading root segment is zero-padded to 2
digits.
EOF
}

discover_cases || exit 2

declare -a SELECTED_CASES=()
declare -A SELECTED_CASE_SET=()
omp_threads="${OMP_NUM_THREADS:-8}"
all_cases=false
force=false
verbose=false

# Zero-pad the leading root segment of a dotted case ID (e.g. "3.2" -> "03.2").
# Prints nothing and returns nonzero if the value isn't a dotted numeric ID.
normalize_case_id() {
  local raw="$1" root rest
  root="${raw%%.*}"
  [[ "${raw}" == *.* ]] && rest="${raw#*.}" || rest=""
  [[ "${root}" =~ ^[0-9]+$ ]] || return 1
  [[ -z "${rest}" ]] || [[ "${rest}" =~ ^[0-9]+(\.[0-9]+)*$ ]] || return 1
  root="$(printf '%02d' "$((10#${root}))")"
  if [[ -n "${rest}" ]]; then
    printf '%s.%s' "${root}" "${rest}"
  else
    printf '%s' "${root}"
  fi
}

append_cases() {
  local value normalized case_number found
  IFS=',' read -r -a requested <<< "$1"
  for value in "${requested[@]}"; do
    normalized="$(normalize_case_id "${value}")" || {
      echo "Invalid case number: ${value}" >&2
      return 2
    }
    if [[ -n "${CASE_CONFIGS[${normalized}]+x}" ]]; then
      # Exact case ID match -- select only that case, even if it has sub-cases.
      if [[ -z "${SELECTED_CASE_SET[${normalized}]+x}" ]]; then
        SELECTED_CASES+=("${normalized}")
        SELECTED_CASE_SET["${normalized}"]=1
      fi
      continue
    fi
    # Otherwise treat it as a group prefix: expand to every descendant case.
    found=false
    for case_number in "${CASE_ORDER[@]}"; do
      if [[ "${case_number}" == "${normalized}."* ]]; then
        if [[ -z "${SELECTED_CASE_SET[${case_number}]+x}" ]]; then
          SELECTED_CASES+=("${case_number}")
          SELECTED_CASE_SET["${case_number}"]=1
        fi
        found=true
      fi
    done
    if [[ "${found}" != true ]]; then
      echo "Unknown case: ${value}" >&2
      return 2
    fi
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
    --force)
      force=true
      shift
      ;;
    --verbose)
      verbose=true
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
  SELECTED_CASES=("${CASE_ORDER[@]}")
fi

if [[ ${#SELECTED_CASES[@]} -eq 0 ]]; then
  echo "Select cases with --cases LIST or use --all." >&2
  usage >&2
  exit 2
fi

export OMP_NUM_THREADS="${omp_threads}"
export PYTHONUNBUFFERED=1

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  COLOR_BOLD=$'\033[1m'
  COLOR_GREEN=$'\033[32m'
  COLOR_RED=$'\033[31m'
  COLOR_YELLOW=$'\033[33m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_BOLD=""
  COLOR_GREEN=""
  COLOR_RED=""
  COLOR_YELLOW=""
  COLOR_RESET=""
fi

status_line() {
  printf '%s%s%s\n' "$1" "$2" "${COLOR_RESET}"
}
BATCH_ID="selected_validation_$(date -u +%Y%m%d_%H%M%S)"
BATCH_DIR="${REPO_ROOT}/scripts/benchmark_results/${BATCH_ID}"
mkdir -p "${BATCH_DIR}"

overall_start=$(date +%s)
failed=0
skipped=0
echo "Running cases: ${SELECTED_CASES[*]}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "Batch directory: ${BATCH_DIR}"

for case_number in "${SELECTED_CASES[@]}"; do
  config_file="${CASE_CONFIGS[${case_number}]}"
  config_path="${CONFIG_DIR}/${config_file}"
  log_file="${BATCH_DIR}/case_${case_number}.log"
  case_start=$(date +%s)

  echo
  status_line "${COLOR_BOLD}" "[${case_number}] ${config_file}"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config: ${config_path}" | tee "${log_file}"
    status=2
  else
    if [[ "${force}" != true ]]; then
      check_output="$(
        uv run python "${REPO_ROOT}/scripts/render_validation_dashboard.py" \
          --check-completed "${case_number}" 2>&1
      )"
      check_status=$?
      if [[ ${check_status} -eq 0 ]]; then
        echo "${check_output}" | tee "${log_file}"
        echo "Case ${case_number}: SKIPPED (unchanged completed result)"
        skipped=$((skipped + 1))
        if [[ "${check_output}" == *"gate=FAIL"* ]]; then
          failed=1
        fi
        continue
      fi
      echo "${check_output}"
    fi
    status_line "${COLOR_YELLOW}" "[${case_number}] running (details: ${log_file})"
    if [[ "${verbose}" == true ]]; then
      bash "${REPO_ROOT}/scripts/compare_backends_report.sh" \
        --config "${config_path}" 2>&1 | tee "${log_file}"
      status="${PIPESTATUS[0]}"
    else
      bash "${REPO_ROOT}/scripts/compare_backends_report.sh" \
        --config "${config_path}" >"${log_file}" 2>&1
      status=$?
    fi
  fi

  elapsed=$(( $(date +%s) - case_start ))
  if [[ ${status} -eq 0 ]]; then
    status_line "${COLOR_GREEN}" "[${case_number}] PASS (${elapsed}s)"
  else
    status_line "${COLOR_RED}" "[${case_number}] FAIL (exit ${status}, ${elapsed}s)"
    failed=1
  fi
done

overall_elapsed=$(( $(date +%s) - overall_start ))
echo
echo "Selected validation run finished in ${overall_elapsed}s."
echo "Skipped unchanged completed cases: ${skipped}"
echo "Logs and batch artifacts: ${BATCH_DIR}"

if uv run python "${REPO_ROOT}/scripts/render_validation_dashboard.py"; then
  echo "Validation dashboard: ${REPO_ROOT}/scripts/benchmark_results/dashboard/index.html"
else
  echo "WARNING: validation dashboard generation failed." >&2
  failed=1
fi

exit "${failed}"
