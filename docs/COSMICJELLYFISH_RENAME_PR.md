# Rename PR: cosmicfishpie -> cosmicjellyfish

Working document for the rename PR. Updated as ideas and questions come in.
Companion to `COSMICJELLYFISH_RELEASE_NOTES.md` (overall release checklist).

Last updated: 2026-11-03

## 1. State of the branch

- Worktree: `cfp-rename`, branch `rename/cosmicjellyfish`.
- Base: `4b44649`, which is contained in `main` (PR #231 was merged as merge commit `aed0d62`
  on 2026-11-03). The PR can target `main` directly; no rebase is needed.
- Commits:
  1. `a70abc5` Rename the package to cosmicjellyfish (292 files, 89 renames).
  2. `2273382` Keep the cosmicfishpie import path as a deprecated alias.
  3. (pending commit) `pr_checks.yml` also watches `cosmicfishpie/**`, so the CHANGELOG check
     runs for PRs that only touch the alias.
- Nothing from this branch has been pushed yet.

### Validation done on the first two commits
- Full suite with the `jaxcosmo` extra (Python 3.12, NumPy 2.2.6): 546 passed, 3 skipped, 73% coverage.
- Without the extra: 451 passed, 8 skipped.
- Strict Sphinx build (`-W`), isort, Black, Ruff, `uv lock --check`, `git diff --check`: clean.
- Wheel and sdist built and installed into clean environments outside the checkout: both
  `cosmicjellyfish` and the `cosmicfishpie` alias import, package data present.
- Spawn multiprocessing check passes.
- Notebook audit: 27 notebooks changed, only code-cell sources, no outputs or metadata.

## 2. Review findings

### Intentional remaining mentions of the old name
- `chains/README.md`, `sampler_scripts/README.md`: external repo `cosmicfishpie-validation`.
- `cosmicjellyfish/cosmology/cosmology.py:974`: `neutrino_scheme` accepts `"cosmicjellyfish"` and the legacy `"cosmicfishpie"`.
- `cosmicjellyfish/likelihood/sampler.py:377`: sampler metadata key `cosmicfishpie_version`
  (kept so tools reading old chains keep working; asserted in `tests/sampler_test.py:738`).
- `docs/source/migration_cosmicjellyfish.md` and `tests/legacy_import_test.py`.
- Historical files, left untouched: `CHANGELOG.md`, `docs/CONFIG_DIAGNOSIS.md`,
  `docs/EUCLID_VALIDATION_FINDINGS.md`, `docs/KNOWN_ISSUES.md`, `docs/VALIDATION_SUMMARY.md`,
  `docs/source/_static/ns_response/ns_response_metadata.json`.
- Environment variables (`CFP_DATA_REPO`, `COSMICFISH_FAST_*`) were not renamed.

### Fixed
- `.github/workflows/pr_checks.yml`: the path filter only had `cosmicjellyfish/**`; added `cosmicfishpie/**`.

### Things to be aware of
1. Links point to `github.com/santiagocasas/cosmicjellyfish`, which only exists after the repo rename.
2. GitHub Pages does not redirect `/cosmicfishpie/...` (see section 4).
3. The alias docstring says "one release cycle", but the support period is still an open decision.
4. Both distributions provide a top-level `cosmicfishpie`; users should uninstall the old one
   before installing `cosmicjellyfish`. TODO: add one line to `migration_cosmicjellyfish.md`.
5. Pre-existing quirks, not caused by the rename:
   - the built package lists `notebooks` and `sampler_scripts` in `top_level.txt`;
   - `install_external_data.py` download URLs point at a `survey_specifications/` folder that
     does not exist in the repo.

## 3. Publishing order (commit and push are done by the user)

1. Commit the `pr_checks.yml` change on `rename/cosmicjellyfish`.
2. `git push -u origin HEAD:rename/cosmicjellyfish`.
3. Create the PR as a draft against `main` with `gh pr create`, using the body in section 5.
4. Add the remaining items (logo, badges, two folders, TestPyPI) as further commits.

## 4. Moving the GitHub Pages site

Today:
- `www.santicasas.space` is the custom domain of the user-site repo `santiagocasas/santiagocasas.github.io`
  (default branch `master`, Pages source `gh-pages`, al-folio/Jekyll site).
- `cosmicfishpie` is a project site served at `/cosmicfishpie/` from its own `gh-pages` branch
  (`docs/`, `dashboard/`, `index.html`, `.nojekyll`; no `CNAME` file, `build_type: legacy`).

On rename:
- `gh-pages` moves with the repo, so content appears at `/cosmicjellyfish/...` at once.
- Repo URLs and git remotes are redirected by GitHub. Pages URLs are not: `/cosmicfishpie/...` becomes 404.

Recommended order:
1. Rename the repository (Settings -> General, or `gh repo rename cosmicjellyfish`). Do it before
   merging the PR so `docs.yml` publishes to the new URL straight away.
2. Update every local clone and worktree, including the workstation:
   `git remote set-url origin git@github.com:santiagocasas/cosmicjellyfish.git`.
3. Merge the PR; `docs.yml` publishes to `/cosmicjellyfish/docs/`.
4. Add redirect stubs to `santiagocasas.github.io`. Once the project site no longer claims
   `/cosmicfishpie/`, files in that repo's `cosmicfishpie/` folder are served. Edit the Jekyll
   source, not its `gh-pages` branch (the al-folio deploy overwrites it).
   - `cosmicfishpie/index.html`, `cosmicfishpie/docs/index.html`, `cosmicfishpie/dashboard/index.html`,
     each with `<meta http-equiv="refresh" content="0; url=/cosmicjellyfish/...">` and a
     `<link rel="canonical">`.
   - A small JavaScript snippet in `404.html` rewriting `/cosmicfishpie/<path>` to `/cosmicjellyfish/<path>`,
     so deep links to individual docs pages keep working.
   - Not yet checked: whether al-folio's `_config.yml` `exclude`/`include` settings drop these paths.
5. Do not create a new repo called `cosmicfishpie` to host redirects: that would break GitHub's
   automatic redirect from the old repo URL.
6. After the rename, re-check:
   - Codecov repo link and token.
   - DeepWiki badge.
   - External links to the docs or dashboard (papers, other READMEs, `cjfp-data`).
7. Register PyPI/TestPyPI Trusted Publishing after the rename, with the final repo name,
   workflow filename and environment name.

Open question: should the redirect stubs be written in the user-site repo by the assistant? (Separate repo; needs an explicit OK.)

## 5. PR text (draft)

Title: **Rename the package to cosmicjellyfish and keep `cosmicfishpie` as a deprecated alias**

The body is kept in `/tmp/opencode/pr_rename_body.md` while drafting and is reproduced here.

```markdown
## Summary
Renames the distribution and the import package from `cosmicfishpie` to `cosmicjellyfish`.
Existing code keeps working through a deprecated `cosmicfishpie` alias.

Based on `main` after #231 (merge commit `aed0d62`).

### Commits so far
1. **Rename the package to cosmicjellyfish** (`a70abc5`): `git mv cosmicfishpie cosmicjellyfish` plus a
   scripted replacement across code, tests, scripts, docs, workflows and packaging (`pyproject.toml`,
   `uv.lock`, `MANIFEST.in`, `.coveragerc`, `pytest.ini`, `Makefile`, `CITATION.cff`,
   `.all-contributorsrc`). In notebooks only code cells were changed; outputs, metadata and prose are untouched.
2. **Keep the cosmicfishpie import path as a deprecated alias** (`2273382`): a small `cosmicfishpie`
   package that emits a `DeprecationWarning` and redirects every `cosmicfishpie.*` import to the *same*
   module object in `cosmicjellyfish.*` (no second copy, so config globals, class identity and pickles
   stay consistent). Adds `docs/source/migration_cosmicjellyfish.md`.
3. **PR checks also watch the alias**: the CHANGELOG check now triggers on `cosmicfishpie/**` as well.

### Deliberately preserved
- `CHANGELOG.md` history and historical docs and stored data.
- Sampler metadata key `cosmicfishpie_version`.
- CLASS `neutrino_scheme: cosmicfishpie` is still accepted; the new default is `cosmicjellyfish`.
- The external repository name `cosmicfishpie-validation`, and environment variables (`CFP_*`, `COSMICFISH_*`).

## Still to be added in this PR
- [ ] New logo (README, docs, favicon)
- [ ] Fix README / docs badges
- [ ] Reorganise two more folders (to be specified)
- [ ] TestPyPI: release workflow with Trusted Publishing, rehearsal upload

## Testing
(see section 1 of this document)

## Before / after merge
- Rename the GitHub repository to `cosmicjellyfish`.
- GitHub Pages URLs under `/cosmicfishpie/` do not redirect; redirect stubs go into the user-site repo.
- Uninstall any existing `cosmicfishpie` before installing `cosmicjellyfish`.
- Open decisions: alias support period, first version number, replacing the direct-URL
  `symbolic-pofk` dependency before publishing to PyPI.
```

## 6. Planned additions to this PR

| Item | Status | Notes / questions |
| --- | --- | --- |
| New logo | open | Need the file(s). Places: README header image, docs `conf.py` (`html_logo`, favicon), `docs/source/_static`, possibly `CITATION.cff`/social preview. |
| Badges | open | README badges (docs, codecov, CI, DeepWiki) use the new repo URL but only resolve after the repo rename. Add PyPI/TestPyPI version and Python-version badges when published. Codecov token may need refreshing. |
| Reorganise two folders | open | Which folders, and where should they go? Note that moving `notebooks/` or `scripts/` touches paths in docs, CI, Makefile and AGENTS.md. |
| TestPyPI | open | Release workflow with Trusted Publishing on a `testpypi` environment; fix `main.yml` release job (python/twine outside uv env), tag-version check, `twine check`; blocker: `symbolic-pofk` direct URL at `pyproject.toml:46`; harden `scripts/maintenance/release/release.sh` (`git add -A`, pushes all tags). |

## 7. Open decisions

- Alias support period (docstring currently says "one release cycle").
- Pickle/checkpoint policy for old artifacts (alias covers `cosmicfishpie.*` class paths; confirm).
- First version number to publish.
- `symbolic-pofk` dependency: replace the direct tarball URL before PyPI.
- Whether the assistant may write the redirect stubs in `santiagocasas.github.io`.

## 8. Log

- 2026-11-03: review of `a70abc5` + `2273382`; found the `pr_checks.yml` gap and fixed it (uncommitted);
  PR body drafted; Pages migration plan written.
