#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# User-configurable settings
RUN_LABEL="validate_many"
RESUME="${RESUME:-true}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-}"
VALIDATION_CONFIGS_DIR="${SCRIPT_DIR}/validation_configs"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_NUM_THREADS
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONUNBUFFERED

if [[ -z "${RUN_ROOT}" && "${RESUME}" == "true" ]]; then
  RUN_ROOT=$(REPO_ROOT="${REPO_ROOT}" python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["REPO_ROOT"]) / "scripts" / "benchmark_results"
label = "validate_many_"
if not root.exists():
    print("")
    raise SystemExit(0)
candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(label)]
if not candidates:
    print("")
    raise SystemExit(0)
latest = max(candidates, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
  )
fi

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${REPO_ROOT}/scripts/benchmark_results/${RUN_LABEL}_${RUN_ID}"
  echo "[validate-many] Starting new run: ${RUN_ROOT}"
else
  echo "[validate-many] Resuming run: ${RUN_ROOT}"
fi

COMBINED_REPORT="${RUN_ROOT}/report_many_${RUN_ID}.html"
INDEX_DIR="${RUN_ROOT}/index"

echo "[validate-many] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[validate-many] PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"

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
  report_folders=()
  for label in "${RUN_LABELS[@]}"; do
    report_folders+=("${RUN_ROOT}/${label}")
  done
  uv run python "${SCRIPT_DIR}/render_compare_reports.py" \
    "${report_folders[@]}" \
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
