#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# User-configurable settings
RUN_LABEL="validate_many"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT}/scripts/benchmark_results/${RUN_LABEL}_${RUN_ID}"
COMBINED_REPORT="${RUN_ROOT}/report_many_${RUN_ID}.html"
INDEX_DIR="${RUN_ROOT}/index"
VALIDATION_CONFIGS_DIR="${SCRIPT_DIR}/validation_configs"

CONFIG_FILES=(
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_01_class_camb_photo_mpvalidation_w0waCDM"
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_02_class_camb_spectro_mpvalidation_w0waCDM"
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_03_camb_class_photo_nuvalidation_nuCDM"
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_04_camb_class_spectro_nuvalidation_nuCDM"
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_05_symbolic_camb_spectro_default_LCDM"
  "${VALIDATION_CONFIGS_DIR}/compare_run_config.env_06_symbolic_camb_photo_default_LCDM"
)

RUN_LABELS=(
  "01_class_camb_photo_mpvalidation_w0waCDM"
  "02_class_camb_spectro_mpvalidation_w0waCDM"
  "03_camb_class_photo_nuvalidation_nuCDM"
  "04_camb_class_spectro_nuvalidation_nuCDM"
  "05_symbolic_camb_spectro_default_LCDM"
  "06_symbolic_camb_photo_default_LCDM"
)

if [[ ${#CONFIG_FILES[@]} -ne ${#RUN_LABELS[@]} ]]; then
  echo "CONFIG_FILES and RUN_LABELS length mismatch" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

is_complete() {
  local dir="$1"
  if [[ ! -f "${dir}/report_single.html" ]]; then
    return 1
  fi
  if ! compgen -G "${dir}/compare_fishers_*.json" >/dev/null; then
    return 1
  fi
  return 0
}

successes=()
failures=()
skipped=()

echo "[validate-many] Root: ${RUN_ROOT}"
for i in "${!CONFIG_FILES[@]}"; do
  cfg="${CONFIG_FILES[$i]}"
  label="${RUN_LABELS[$i]}"
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" >&2
    failures+=("${label}")
    continue
  fi
  outdir="${RUN_ROOT}/${label}"
  if is_complete "${outdir}"; then
    echo "[validate-many] Skipping ${label} (already complete)"
    skipped+=("${label}")
    continue
  fi
  echo "[validate-many] Running ${label}"
  set +e
  OUTDIR="${outdir}" bash "${SCRIPT_DIR}/compare_backends_report.sh" --config "${cfg}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[validate-many][WARN] ${label} failed (exit ${rc})" >&2
    failures+=("${label}")
    continue
  fi
  if is_complete "${outdir}"; then
    successes+=("${label}")
  else
    echo "[validate-many][WARN] ${label} did not produce expected outputs" >&2
    failures+=("${label}")
  fi
done

if [[ ${#successes[@]} -gt 0 || ${#skipped[@]} -gt 0 ]]; then
  echo "[validate-many] Building combined report"
  uv run python "${SCRIPT_DIR}/render_compare_reports.py" \
    --glob "${RUN_ROOT}/*" \
    --index-dir "${INDEX_DIR}" \
    --single-file "${COMBINED_REPORT}"
else
  echo "[validate-many][WARN] No successful runs; skipping combined report" >&2
fi

if [[ ${#successes[@]} -gt 0 ]]; then
  echo "[validate-many] Success: ${successes[*]}"
fi
if [[ ${#skipped[@]} -gt 0 ]]; then
  echo "[validate-many] Skipped: ${skipped[*]}"
fi
if [[ ${#failures[@]} -gt 0 ]]; then
  echo "[validate-many] Failed: ${failures[*]}" >&2
fi

if [[ ${#successes[@]} -gt 0 || ${#skipped[@]} -gt 0 ]]; then
  exit 0
fi
exit 1
