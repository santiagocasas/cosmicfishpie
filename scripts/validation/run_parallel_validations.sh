#!/usr/bin/env bash
# Parallel convenience entry point: two Fisher validation cases at a time,
# with four OpenMP threads per case (eight threads total).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_REPO="${CFP_DATA_REPO:-${PROJECT_ROOT}/../cjfp-data}"
if ! git -C "${DATA_REPO}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Validation data repository not found: ${DATA_REPO}" >&2
  echo "Clone cjfp-data beside the CosmicFishPie checkout, or set CFP_DATA_REPO." >&2
  exit 2
fi
export CFP_VALIDATION_RESULTS_DIR="${CFP_VALIDATION_RESULTS_DIR:-${DATA_REPO}/validation/fisher}"

OMP_THREADS="${CFP_OMP_THREADS:-4}"
JOBS="${CFP_VALIDATION_JOBS:-2}"

args=("$@")
has_selection=false
for arg in "${args[@]}"; do
  case "${arg}" in
    --all|--cases|--cases=*) has_selection=true ;;
  esac
done
if [[ "${has_selection}" != true ]]; then
  args+=(--all)
fi

exec bash "${SCRIPT_DIR}/run_selected_validations.sh" \
  --omp-threads "${OMP_THREADS}" \
  --jobs "${JOBS}" \
  "${args[@]}"
