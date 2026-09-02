#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CFP_ROOT="$ROOT"

PLANCK_DIR_DEFAULT="$ROOT/Planck-Results/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00/base/plikHM_TTTEEE_lowl_lowE"
PLANCK_DIR="${1:-$PLANCK_DIR_DEFAULT}"
CHAIN_ROOT="base_plikHM_TTTEEE_lowl_lowE"

REQUIRED_FILES=(
  "$PLANCK_DIR/dist/${CHAIN_ROOT}.likestats"
  "$PLANCK_DIR/dist/${CHAIN_ROOT}.margestats"
  "$PLANCK_DIR/dist/${CHAIN_ROOT}.covmat"
  "$PLANCK_DIR/${CHAIN_ROOT}.paramnames"
  "$PLANCK_DIR/${CHAIN_ROOT}.minimum.inputparams"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    printf 'Missing required Planck file: %s\n' "$file"
    printf 'Download Planck products from:\n'
    printf 'https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters\n'
    exit 1
  fi
done

echo "== Legacy full (2..2508, T+E) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_T,CMB_E \
  --lmin 2 \
  --lmax 2508 \
  --cmb-noise-model legacy \
  --outdir "$ROOT/tmp/planck_bestfit_theta"

echo "== Knox full (2..2508, T+E) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_T,CMB_E \
  --lmin 2 \
  --lmax 2508 \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir "$ROOT/tmp/planck_bestfit_theta_knox_full"

echo "== Knox full lmax=1500 (2..1500, T+E) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_T,CMB_E \
  --lmin 2 \
  --lmax 1500 \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir "$ROOT/tmp/planck_bestfit_theta_knox_full_l1500"

echo "== Knox 3-part: high (30..2508, T+E) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_T,CMB_E \
  --lmin 30 \
  --lmax 2508 \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir "$ROOT/tmp/planck_bestfit_theta_knox_3part/high_te"

echo "== Knox 3-part: low TT (2..29) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_T \
  --lmin 2 \
  --lmax 29 \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir "$ROOT/tmp/planck_bestfit_theta_knox_3part/low_t"

echo "== Knox 3-part: low EE (2..29) =="
uv run python "$ROOT/scripts/run_planck_bestfit_fisher.py" \
  --planck-dir "$PLANCK_DIR" \
  --chain-root "$CHAIN_ROOT" \
  --parameterization theta \
  --observables CMB_E \
  --lmin 2 \
  --lmax 29 \
  --cmb-noise-model knox \
  --ee-lowell-noise-boost 8 \
  --ee-lowell-max-ell 29 \
  --outdir "$ROOT/tmp/planck_bestfit_theta_knox_3part/low_e"

echo "== Combine 3-part Fisher =="
uv run python - <<'PY'
import os
from pathlib import Path

from cosmicfishpie.analysis import fisher_matrix as fm

root = Path(os.environ["CFP_ROOT"])
base = root / "tmp/planck_bestfit_theta_knox_3part"

high = fm.fisher_matrix(
    file_name=str(base / "high_te/CosmicFish_v1.3.0_planck_bestfit_theta_primary__CMB_TCMB_E_FM.txt")
)
low_t = fm.fisher_matrix(
    file_name=str(base / "low_t/CosmicFish_v1.3.0_planck_bestfit_theta_primary__CMB_T_FM.txt")
)
low_e = fm.fisher_matrix(
    file_name=str(base / "low_e/CosmicFish_v1.3.0_planck_bestfit_theta_primary__CMB_E_FM.txt")
)

combined = high + low_t + low_e
combined.name = "CosmicFish_v1.3.0_planck_bestfit_theta_knox_3part_FM"

outdir = base / "combined"
outdir.mkdir(parents=True, exist_ok=True)
outroot = outdir / combined.name
combined.save_to_file(str(outroot))
print(f"Wrote {outroot}.txt")
PY

echo "Done. Notebook diagnostics inputs are ready under tmp/."
