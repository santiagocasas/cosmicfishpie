#!/usr/bin/env bash
set -euo pipefail

# Publish the locally generated dashboard without committing benchmark output
# to the source branch. Existing gh-pages content outside dashboard/ is kept.

repo_root=$(git rev-parse --show-toplevel)
dashboard_dir="${DASHBOARD_DIR:-$repo_root/scripts/benchmark_results/dashboard}"
branch="${PAGES_BRANCH:-gh-pages}"
remote="${PAGES_REMOTE:-origin}"
target_dir="${PAGES_SUBPATH:-dashboard}"

if [[ ! -f "$dashboard_dir/index.html" ]]; then
    printf 'Dashboard not found: %s/index.html\n' "$dashboard_dir" >&2
    printf 'Run: uv run python scripts/render_validation_dashboard.py\n' >&2
    exit 1
fi

worktree=$(mktemp -d "${TMPDIR:-/tmp}/cosmicfishpie-pages.XXXXXX")
cleanup() {
    git worktree remove --force "$worktree" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if git show-ref --verify --quiet "refs/heads/$branch"; then
    git worktree add "$worktree" "$branch" >/dev/null
else
    git worktree add --detach "$worktree" HEAD >/dev/null
    git -C "$worktree" checkout --orphan "$branch" >/dev/null
    git -C "$worktree" rm -rf . >/dev/null 2>&1 || true
fi

rm -rf "$worktree/$target_dir"
mkdir -p "$worktree/$target_dir"
cp -R "$dashboard_dir"/. "$worktree/$target_dir"/

git -C "$worktree" add "$target_dir"
if git -C "$worktree" diff --cached --quiet; then
    printf 'Dashboard is already up to date on %s.\n' "$branch"
    exit 0
fi

git -C "$worktree" commit -m "Publish validation dashboard" >/dev/null
git -C "$worktree" push "$remote" "$branch"
printf 'Published dashboard to %s/%s on %s.\n' "$remote" "$branch" "$target_dir"
