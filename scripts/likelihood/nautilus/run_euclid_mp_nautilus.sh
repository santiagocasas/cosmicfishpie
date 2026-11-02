#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec uv run --project "$ROOT" python "$ROOT/scripts/likelihood/nautilus/run_euclid_mp_nautilus.py" "$@"
