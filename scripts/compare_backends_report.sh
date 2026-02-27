#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_OUTDIR="${OUTDIR:-}"

CONFIG_FILE=""
usage() {
  cat <<'EOF'
Usage: compare_backends_report.sh [--config path]

Runs two backends, compares, plots, and generates reports.

Options:
  --config PATH   Path to an env-style config file (KEY=VALUE)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --config=*)
      CONFIG_FILE="${1#*=}"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# User-configurable settings
MODE="photo"              # photo | spectro
CODE_A="class"            # class | camb | symbolic
CODE_B="camb"             # class | camb | symbolic
YAML_A=""                # optional: path to YAML for run A
YAML_B=""                # optional: path to YAML for run B
YAML_KEY_A=""            # optional: override yaml key for run A
YAML_KEY_B=""            # optional: override yaml key for run B
COMMON_SPECS_JSON=""     # optional: JSON with fiducialpars/freepars/options
ACCURACY=1
FEEDBACK=1
OMP_THREADS=""           # optional: set OMP_NUM_THREADS
FOM_PARAMS="Omegam,sigma8"
PLOT=true                # set false to skip plot generation
USE_TIMESTAMP=false      # set true to append a timestamp to outputs
OUTDIR=""                # optional: output directory; default uses config hash

REPORT_SINGLE=true       # set false to skip single-file report
REPORT_SINGLE_FILE=""    # optional: single-file HTML path; default is OUTDIR/report_single.html
INDEX_DIR=""             # optional: index directory; default is OUTDIR/index
BUNDLE_DIR=""            # optional: set to generate a share bundle
MAKE_ZIP=false           # set true to zip the bundle

if [[ -n "${CONFIG_FILE}" ]]; then
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

if [[ -n "${ENV_OUTDIR}" && -z "${OUTDIR}" ]]; then
  OUTDIR="${ENV_OUTDIR}"
fi

if [[ "${REPORT_SINGLE_FILE}" == "true" || "${REPORT_SINGLE_FILE}" == "false" ]]; then
  echo "[config][WARN] REPORT_SINGLE_FILE expects a path. Interpreting '${REPORT_SINGLE_FILE}' as REPORT_SINGLE." >&2
  REPORT_SINGLE="${REPORT_SINGLE_FILE}"
  REPORT_SINGLE_FILE=""
fi

if [[ "${REPORT_SINGLE}" != "true" && "${REPORT_SINGLE}" != "false" ]]; then
  echo "Invalid REPORT_SINGLE value: ${REPORT_SINGLE} (expected true/false)" >&2
  exit 2
fi
if [[ "${REPORT_SINGLE}" == "false" && -n "${REPORT_SINGLE_FILE}" ]]; then
  echo "[config][WARN] REPORT_SINGLE=false; ignoring REPORT_SINGLE_FILE='${REPORT_SINGLE_FILE}'." >&2
  REPORT_SINGLE_FILE=""
fi

if [[ -n "${COMMON_SPECS_JSON}" && "${COMMON_SPECS_JSON}" != /* ]]; then
  if [[ -n "${CONFIG_FILE}" ]]; then
    COMMON_SPECS_JSON="$(cd -- "$(dirname -- "${CONFIG_FILE}")" && pwd)/${COMMON_SPECS_JSON}"
  else
    COMMON_SPECS_JSON="${REPO_ROOT}/${COMMON_SPECS_JSON}"
  fi
fi
if [[ -n "${COMMON_SPECS_JSON}" && ! -f "${COMMON_SPECS_JSON}" ]]; then
  echo "COMMON_SPECS_JSON not found: ${COMMON_SPECS_JSON}" >&2
  exit 2
fi

_default_yaml() {
  local code="$1"
  local defaults_dir="${REPO_ROOT}/cosmicfishpie/configs/default_boltzmann_yaml_files"
  case "${code}" in
    class)
      echo "${defaults_dir}/class/default.yaml"
      ;;
    camb)
      echo "${defaults_dir}/camb/default.yaml"
      ;;
    symbolic)
      echo "${defaults_dir}/symbolic/default.yaml"
      ;;
    *)
      echo ""
      ;;
  esac
}

_abs_path() {
  TARGET_PATH="$1" python - <<'PY'
from pathlib import Path
import os

p = Path(os.environ["TARGET_PATH"]).expanduser()
try:
    resolved = p.resolve(strict=False)
except TypeError:
    resolved = p.resolve()
print(resolved)
PY
}

APPLIED_DEFAULT_YAML_A=false
APPLIED_DEFAULT_YAML_B=false
if [[ -z "${YAML_A}" ]]; then
  YAML_A="$(_default_yaml "${CODE_A}")"
  APPLIED_DEFAULT_YAML_A=true
fi
if [[ -z "${YAML_B}" ]]; then
  YAML_B="$(_default_yaml "${CODE_B}")"
  APPLIED_DEFAULT_YAML_B=true
fi

if [[ -z "${YAML_A}" ]]; then
  echo "No default YAML for code-a='${CODE_A}'. Set YAML_A in the config." >&2
  exit 2
fi
if [[ -z "${YAML_B}" ]]; then
  echo "No default YAML for code-b='${CODE_B}'. Set YAML_B in the config." >&2
  exit 2
fi
if [[ ! -f "${YAML_A}" ]]; then
  echo "YAML_A not found: ${YAML_A}" >&2
  exit 2
fi
if [[ ! -f "${YAML_B}" ]]; then
  echo "YAML_B not found: ${YAML_B}" >&2
  exit 2
fi
if [[ "${APPLIED_DEFAULT_YAML_A}" == "true" ]]; then
  echo "[config] YAML_A not set; using default: ${YAML_A}"
fi
if [[ "${APPLIED_DEFAULT_YAML_B}" == "true" ]]; then
  echo "[config] YAML_B not set; using default: ${YAML_B}"
fi

COMMON_SPECS_HASH=""
if [[ -n "${COMMON_SPECS_JSON}" ]]; then
  COMMON_SPECS_HASH=$(
    COMMON_SPECS_JSON="${COMMON_SPECS_JSON}" python - <<'PY'
import hashlib
import os

path = os.environ["COMMON_SPECS_JSON"]
with open(path, "rb") as fh:
    data = fh.read()
print(hashlib.sha256(data).hexdigest())
PY
  )
fi

CONFIG_STRING="MODE=${MODE}
CODE_A=${CODE_A}
CODE_B=${CODE_B}
YAML_A=${YAML_A}
YAML_B=${YAML_B}
YAML_KEY_A=${YAML_KEY_A}
YAML_KEY_B=${YAML_KEY_B}
COMMON_SPECS_JSON=${COMMON_SPECS_JSON}
COMMON_SPECS_HASH=${COMMON_SPECS_HASH}
ACCURACY=${ACCURACY}
FEEDBACK=${FEEDBACK}
OMP_THREADS=${OMP_THREADS}
FOM_PARAMS=${FOM_PARAMS}
PLOT=${PLOT}"
export CONFIG_STRING
CONFIG_HASH=$(
  python - <<'PY'
import hashlib
import os

cfg = os.environ["CONFIG_STRING"].encode("utf-8")
print(hashlib.sha256(cfg).hexdigest())
PY
)
HASH_SHORT="${CONFIG_HASH:0:10}"

NAME_SUFFIX="cfg_${HASH_SHORT}"
if [[ "${USE_TIMESTAMP}" == "true" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  NAME_SUFFIX="${NAME_SUFFIX}_${RUN_ID}"
fi

if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="${REPO_ROOT}/scripts/benchmark_results/compare_${MODE}_${CODE_A}_vs_${CODE_B}_${NAME_SUFFIX}"
fi
if [[ "${REPORT_SINGLE}" == "true" && -z "${REPORT_SINGLE_FILE}" ]]; then
  REPORT_SINGLE_FILE="${OUTDIR}/report_single.html"
fi
if [[ -z "${INDEX_DIR}" ]]; then
  INDEX_DIR="${OUTDIR}/index"
fi
if [[ -n "${REPORT_SINGLE_FILE}" && "${REPORT_SINGLE_FILE}" != /* ]]; then
  REPORT_SINGLE_FILE="${OUTDIR}/${REPORT_SINGLE_FILE}"
fi
if [[ -n "${INDEX_DIR}" && "${INDEX_DIR}" != /* ]]; then
  INDEX_DIR="${OUTDIR}/${INDEX_DIR}"
fi
if [[ -n "${BUNDLE_DIR}" && "${BUNDLE_DIR}" != /* ]]; then
  BUNDLE_DIR="${OUTDIR}/${BUNDLE_DIR}"
fi

OUTDIR_ABS="$(_abs_path "${OUTDIR}")"
if [[ -n "${REPORT_SINGLE_FILE}" ]]; then
  REPORT_SINGLE_ABS="$(_abs_path "${REPORT_SINGLE_FILE}")"
  case "${REPORT_SINGLE_ABS}" in
    "${OUTDIR_ABS}"/*) ;;
    *)
      echo "REPORT_SINGLE_FILE must be inside OUTDIR: ${REPORT_SINGLE_ABS}" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${INDEX_DIR}" ]]; then
  INDEX_DIR_ABS="$(_abs_path "${INDEX_DIR}")"
  case "${INDEX_DIR_ABS}" in
    "${OUTDIR_ABS}"/*) ;;
    *)
      echo "INDEX_DIR must be inside OUTDIR: ${INDEX_DIR_ABS}" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${BUNDLE_DIR}" ]]; then
  BUNDLE_DIR_ABS="$(_abs_path "${BUNDLE_DIR}")"
  case "${BUNDLE_DIR_ABS}" in
    "${OUTDIR_ABS}"/*) ;;
    *)
      echo "BUNDLE_DIR must be inside OUTDIR: ${BUNDLE_DIR_ABS}" >&2
      exit 2
      ;;
  esac
fi

mkdir -p "${OUTDIR}"
RUN_CONFIG_PATH="${OUTDIR}/run_config.env"
COMMON_SPECS_COPY=""
if [[ -n "${COMMON_SPECS_JSON}" ]]; then
  COMMON_SPECS_COPY="${OUTDIR}/common_specs.json"
  cp "${COMMON_SPECS_JSON}" "${COMMON_SPECS_COPY}"
fi
{
  printf "MODE=%q\n" "${MODE}"
  printf "CODE_A=%q\n" "${CODE_A}"
  printf "CODE_B=%q\n" "${CODE_B}"
  printf "YAML_A=%q\n" "${YAML_A}"
  printf "YAML_B=%q\n" "${YAML_B}"
  printf "YAML_KEY_A=%q\n" "${YAML_KEY_A}"
  printf "YAML_KEY_B=%q\n" "${YAML_KEY_B}"
  printf "COMMON_SPECS_JSON=%q\n" "${COMMON_SPECS_JSON}"
  printf "COMMON_SPECS_HASH=%q\n" "${COMMON_SPECS_HASH}"
  if [[ -n "${COMMON_SPECS_COPY}" ]]; then
    printf "COMMON_SPECS_COPY=%q\n" "${COMMON_SPECS_COPY}"
  fi
  printf "ACCURACY=%q\n" "${ACCURACY}"
  printf "FEEDBACK=%q\n" "${FEEDBACK}"
  printf "OMP_THREADS=%q\n" "${OMP_THREADS}"
  printf "FOM_PARAMS=%q\n" "${FOM_PARAMS}"
  printf "PLOT=%q\n" "${PLOT}"
  printf "USE_TIMESTAMP=%q\n" "${USE_TIMESTAMP}"
  printf "REPORT_SINGLE=%q\n" "${REPORT_SINGLE}"
  printf "REPORT_SINGLE_FILE=%q\n" "${REPORT_SINGLE_FILE}"
  if [[ -n "${CONFIG_FILE}" ]]; then
    printf "CONFIG_SOURCE=%q\n" "${CONFIG_FILE}"
  fi
  printf "CONFIG_HASH=%q\n" "${CONFIG_HASH}"
  printf "NAME_SUFFIX=%q\n" "${NAME_SUFFIX}"
  printf "OUTDIR=%q\n" "${OUTDIR}"
  printf "INDEX_DIR=%q\n" "${INDEX_DIR}"
  printf "BUNDLE_DIR=%q\n" "${BUNDLE_DIR}"
  printf "MAKE_ZIP=%q\n" "${MAKE_ZIP}"
} > "${RUN_CONFIG_PATH}"

compare_cmd=(
  uv run python "${SCRIPT_DIR}/run_fisher_compare_backends.py"
  --mode "${MODE}"
  --code-a "${CODE_A}"
  --code-b "${CODE_B}"
  --accuracy "${ACCURACY}"
  --feedback "${FEEDBACK}"
  --fom-params "${FOM_PARAMS}"
  --outdir "${OUTDIR}"
  --compare
)

if [[ "${PLOT}" == "true" ]]; then
  compare_cmd+=(--plot)
fi
if [[ -n "${OMP_THREADS}" ]]; then
  compare_cmd+=(--omp-threads "${OMP_THREADS}")
fi
if [[ -n "${YAML_A}" ]]; then
  compare_cmd+=(--yaml-a "${YAML_A}")
fi
if [[ -n "${YAML_B}" ]]; then
  compare_cmd+=(--yaml-b "${YAML_B}")
fi
if [[ -n "${YAML_KEY_A}" ]]; then
  compare_cmd+=(--yaml-key-a "${YAML_KEY_A}")
fi
if [[ -n "${YAML_KEY_B}" ]]; then
  compare_cmd+=(--yaml-key-b "${YAML_KEY_B}")
fi
if [[ -n "${COMMON_SPECS_JSON}" ]]; then
  compare_cmd+=(--common-specs "${COMMON_SPECS_JSON}")
fi

echo "[compare] Running backends into: ${OUTDIR}"
"${compare_cmd[@]}"

report_cmd=(
  uv run python "${SCRIPT_DIR}/render_compare_reports.py"
  "${OUTDIR}"
  --index-dir "${INDEX_DIR}"
)

if [[ -n "${REPORT_SINGLE_FILE}" ]]; then
  report_cmd+=(--single-file "${REPORT_SINGLE_FILE}")
fi
if [[ -n "${BUNDLE_DIR}" ]]; then
  report_cmd+=(--bundle-dir "${BUNDLE_DIR}")
fi
if [[ "${MAKE_ZIP}" == "true" ]]; then
  report_cmd+=(--zip)
fi

echo "[report] Rendering reports"
"${report_cmd[@]}"
