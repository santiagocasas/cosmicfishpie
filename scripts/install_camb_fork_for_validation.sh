#!/usr/bin/env bash
# Install and pin the Santiago Casas CAMB fork (hard-coded low-tolerance
# Halofit convergence) into this project's venv, for backend validation
# scripts only. Standard cosmicfishpie usage keeps using the pinned PyPI
# camb from pyproject.toml/uv.lock; this script never modifies that pin.
#
# Installs directly from the git commit via a VCS URL (no persistent local
# clone is created or managed by this script).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Commit that produced the validated 0.05-0.93% agreement in
# scripts/run_case01_full7_class_hp_vs_camb.py / _dp_vs_camb.py. This HEAD
# is a merge of upstream cmbant/CAMB master and includes, as an ancestor,
# c52551a4c4cc9bf8d6fa64b4cb829365c89242b5 ("hard coding the halofit
# tolerance to 1e-6").
DEFAULT_PINNED_COMMIT="98fe7b578acb091baaf90ac8127e417f1d7ebc82"
DEFAULT_REMOTE="https://github.com/santiagocasas/CAMB.git"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
DEFAULT_GFORTRAN_ENV_NAME="gfortran-build"

REMOTE="${DEFAULT_REMOTE}"
PINNED_COMMIT="${DEFAULT_PINNED_COMMIT}"
TARGET_PYTHON="${DEFAULT_PYTHON}"
RESTORE=false

read -r -d '' HELP_TEXT <<EOF || true
Install and pin the CAMB fork (hard-coded low-tolerance Halofit) for
CLASS/CAMB backend validation scripts.

Background
----------
The standard PyPI 'camb' wheel only exposes 'halofit_tol_sigma' as a
Python-settable Ini parameter; empirically this does not reach the tight
nonlinear power-spectrum convergence used in the Euclid validation paper
(Casas et al. 2023, arXiv:2303.09451). That paper hard-codes the Halofit
convergence tolerance directly in fortran/halofit.f90 at 1e-6. This script
installs that fork, pinned to a fixed validated commit, directly from git
into the project .venv (via 'uv pip install camb @ git+<remote>@<commit>')
-- overriding (but NOT modifying the pin of) the standard camb==1.6.4 from
pyproject.toml/uv.lock. No local clone directory is created or kept.

This override does not persist: any 'uv run <cmd>' (black, ruff, pytest,
etc.) resyncs the venv back to the pinned PyPI camb. To keep the fork
active for a validation run, invoke scripts directly via the venv python:
    .venv/bin/python scripts/run_case01_full7_class_hp_vs_camb.py
To restore the standard camb pin at any time:
    scripts/install_camb_fork_for_validation.sh --restore
    (equivalent to: uv sync --extra dev)

Usage
-----
  scripts/install_camb_fork_for_validation.sh [options]

Options
  --remote URL      Git remote URL, without a 'git+' prefix (HTTPS works
                     without auth since the fork is public)
                     (default: ${DEFAULT_REMOTE})
  --commit SHA      Commit to pin (default: ${DEFAULT_PINNED_COMMIT})
  --python PATH     Target venv python (default: ${DEFAULT_PYTHON})
  --restore         Run 'uv sync --extra dev' to restore standard camb, then exit
  -h, --help        Show this help

Future TODO (explicitly deferred): syncing/rebasing the fork against
upstream cmbant/CAMB master is not implemented here.
EOF

print_help() { printf '%s\n' "${HELP_TEXT}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --commit) PINNED_COMMIT="$2"; shift 2 ;;
    --python) TARGET_PYTHON="$2"; shift 2 ;;
    --restore) RESTORE=true; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_help; exit 1 ;;
  esac
done

if [[ "${RESTORE}" == "true" ]]; then
  echo "[install-camb-fork] Restoring standard pinned camb via 'uv sync --extra dev'..."
  echo "[install-camb-fork] NOTE: 'uv sync' also removes any packages installed"
  echo "  outside the lockfile (e.g. a manually 'pip install classy'). Reinstall"
  echo "  those separately afterward if needed, e.g.:"
  echo "    uv pip install classy --python ${DEFAULT_PYTHON}"
  (cd "${REPO_ROOT}" && uv sync --extra dev)
  echo "[install-camb-fork] Done. Standard camb from pyproject.toml is now active."
  exit 0
fi

if [[ ! -x "${TARGET_PYTHON}" ]]; then
  echo "ERROR: target python not found or not executable: ${TARGET_PYTHON}" >&2
  echo "Run 'uv sync --extra dev' first to create the project venv." >&2
  exit 1
fi

echo "=== Step 1: Ensure gfortran is available ==="
if command -v gfortran >/dev/null 2>&1; then
  echo "[install-camb-fork] Found gfortran on PATH: $(command -v gfortran)"
else
  echo "[install-camb-fork] gfortran not found on PATH; looking for conda/mamba..."
  CONDA_BIN=""
  if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN=mamba
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=conda
  else
    echo "ERROR: no gfortran and no conda/mamba available to install one." >&2
    echo "Install gfortran manually (e.g. 'sudo apt install gfortran') and re-run." >&2
    exit 1
  fi

  # Resolve gfortran's path via '<conda_bin> run', which works consistently
  # across conda/mamba/micromamba (unlike parsing 'info --base', whose output
  # format differs between conda and mamba).
  EXISTING_GFORTRAN="$("${CONDA_BIN}" run -n "${DEFAULT_GFORTRAN_ENV_NAME}" bash -c 'command -v gfortran' 2>/dev/null || true)"

  if [[ -z "${EXISTING_GFORTRAN}" ]]; then
    echo "[install-camb-fork] Creating isolated '${DEFAULT_GFORTRAN_ENV_NAME}' conda env with gfortran..."
    "${CONDA_BIN}" create -n "${DEFAULT_GFORTRAN_ENV_NAME}" -c conda-forge gfortran -y
    EXISTING_GFORTRAN="$("${CONDA_BIN}" run -n "${DEFAULT_GFORTRAN_ENV_NAME}" bash -c 'command -v gfortran')"
  else
    echo "[install-camb-fork] Reusing existing '${DEFAULT_GFORTRAN_ENV_NAME}' conda env."
  fi

  if [[ -z "${EXISTING_GFORTRAN}" ]]; then
    echo "ERROR: gfortran still not found after creating '${DEFAULT_GFORTRAN_ENV_NAME}'." >&2
    exit 1
  fi
  GFORTRAN_ENV_BIN="$(dirname -- "${EXISTING_GFORTRAN}")"
  export PATH="${GFORTRAN_ENV_BIN}:${PATH}"
  echo "[install-camb-fork] Using gfortran: $(command -v gfortran)"
fi

echo "=== Step 2: Install camb directly from the pinned git commit ==="
INSTALL_SPEC="camb @ git+${REMOTE}@${PINNED_COMMIT}"
echo "[install-camb-fork] ${INSTALL_SPEC}"
echo "[install-camb-fork] Target python: ${TARGET_PYTHON}"
(cd "${REPO_ROOT}" && uv pip install "${INSTALL_SPEC}" --python "${TARGET_PYTHON}")

echo "=== Step 3: Verify ==="
"${TARGET_PYTHON}" - <<PY
import json
import importlib.metadata as im

import camb

print(f"[install-camb-fork] camb.__version__ = {camb.__version__}")
print(f"[install-camb-fork] camb.__file__    = {camb.__file__}")

try:
    raw = im.distribution("camb").read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    vcs = info.get("vcs_info", {})
    print(f"[install-camb-fork] commit = {vcs.get('commit_id', '?')}")
    print(f"[install-camb-fork] remote = {info.get('url', '?')}")
except Exception as exc:  # noqa: BLE001
    print(f"[install-camb-fork] (could not read direct_url.json: {exc})")
PY

cat <<'EOF'

[install-camb-fork] Done. The project venv now uses the CAMB fork.

IMPORTANT: This is a temporary override of the venv, not a persistent pin.
  - Any 'uv run <cmd>' (including black/ruff/pytest) will silently resync
    the venv back to the standard pinned camb from pyproject.toml/uv.lock.
  - To run validation scripts against the fork, invoke them directly via
    the venv python, e.g.:
        .venv/bin/python scripts/run_case01_full7_class_hp_vs_camb.py
  - To explicitly restore the standard camb pin at any time:
        scripts/install_camb_fork_for_validation.sh --restore
    (or just run 'uv sync --extra dev')
EOF
