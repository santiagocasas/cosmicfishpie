#!/usr/bin/env bash
# Six Fisher runs, by default using all 36 Freyja logical CPUs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/validation/run_euclid_mp_fishers.py" "$@"
