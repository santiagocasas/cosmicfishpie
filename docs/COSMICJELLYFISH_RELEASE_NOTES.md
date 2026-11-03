# CosmicJellyfish: release direction and deferred development

## Researcher decision

Prioritize releasing **CosmicJellyfish** and migrating the CosmicFishPie name to
`cosmicjellyfish`. The “J” can stand for the project's JAX capability.
Accept the current roughly 10% cross-model Fisher agreement for the present
development stage; do not make a new JAX-SYREN package or upstream integration a
prerequisite for the release. Revisit that work later, rather than running several
development projects in parallel.

This records the direction, not an instruction to perform the package rename now.

## Current scientific comparison

The fixed-ns, massless-neutrino Euclid photometric examples give marginalized
`10^9As` errors of 0.0294396 (SYREN) and 0.0264081 (JAXcosmo), a -10.30% gap;
the second fiducial gives -11.31%. “Roughly 10%” is therefore not a strict
below-10% acceptance threshold or a statement about every cosmology or probe.

The same-theory JAX AD versus JAX finite-difference Fisher comparison is much
closer: maximum matrix-normalized differences of about 1.8e-5 at both fiducials.
Keep this implementation check distinct from cross-model scientific agreement.

Both power models use Eisenstein–Hu as a starting point, but SYREN-NEW includes
fitted linear-spectrum corrections and its own nonlinear emulator; the current
JAXcosmo route uses Eisenstein–Hu with wiggles plus Halofit. A diagnostic replacing
power inside the JAX observable with SYREN power recovers the SYREN amplitude
error to about 0.08%. The experimental Halofit-on-SYREN-linear comparison is
suggestive, but its tabulated linear spectrum is clamped outside the provider's
k range; do not treat it as a precision causal decomposition without checking
the integration tails and normalization.

Evidence in sibling `../cfp-jaxcosmo/`:

- `docs/source/jaxcosmo.md`
- `scripts/validation/compare_theory_amplitude_response.py`
- `scripts/validation/time_syren_vs_jaxcosmo.py`
- `notebooks/07-JAXcosmo-autodiff-vs-finite-differences.ipynb`
- Commits `022d02a` and `1c5b946`.

## Deferred: differentiable SYREN distribution

Possible approaches, with no choice committed yet:

1. Port SYREN's linear **and nonlinear** expressions to JAX in a fork of
   `symbolic_pofk`, preserving a NumPy reference and comparing values and derivatives.
2. Offer that implementation upstream to `symbolic_pofk` or add an opt-in SYREN
   power option to `jax_cosmo`. Keep existing Eisenstein–Hu behavior compatible.
3. If upstream release timing is unsuitable, consider a separately released,
   appropriately attributed package with a distinct import namespace.

SYREN linear power is `plin_new_emulated`; nonlinear power is
`pnl_new_emulated`. A prior discussion incorrectly described the former as a
nonlinear path. Massive-neutrino terms in SYREN do not automatically provide
massive-neutrino support in the surrounding JAX background/growth implementation.

The local smooth-Halofit upstream patch remains a separate potential contribution,
not a dependency of this release direction.

## SYREN versus JAXcosmo: what agrees and what differs

### The formulas are EH-based, not identical

The measured JAXcosmo route uses `jax_cosmo`'s **wiggle** Eisenstein–Hu transfer
(`eisenhu_osc`), normalized through sigma8, followed by Takahashi Halofit with
the smooth nonlinear-scale solver implemented in CosmicFishPie.

SYREN-NEW's `plin_new_emulated` instead evaluates

```text
P_linear = EH_no_wiggle * D_approximate^2 * exp(logF_fiducial) * R * S
```

Here `get_eisensteinhu_nw` supplies the smooth EH backbone. The Bartlett et al.
fitted `logF_fiducial` correction supplies acoustic structure and broadband
corrections relative to that backbone; `R` and `S` provide additional fitted
corrections. Nonlinear power then comes from `pnl_new_emulated`, rather than
Halofit. A shared EH origin therefore does not imply equal final spectra.

**No separate full no-wiggle-versus-wiggle Fisher campaign was performed.**
The comparisons below are corrected SYREN-NEW versus the default JAX EH-wiggle
route. They do not isolate wiggles as the sole cause, nor establish which model
is more accurate against a Boltzmann solver or simulations.

### Linear power and growth

At the first fiducial (`10^9As=2.1`, `Omegam=0.32`, `Omegab=0.05`, `h=0.67`,
`ns=0.96`, `mnu=0`), band-mean ratios at z=0 were:

| k band [h/Mpc] | SYREN linear / JAX EH-wiggle linear |
| --- | ---: |
| 0.001–0.005 | 1.00320 |
| 0.005–0.02 | 1.01637 |
| 0.02–0.05 | 1.00926 |
| 0.05–0.1 | 0.98717 |
| 0.1–0.2 | 0.99135 |
| 0.2–0.5 | 1.00927 |
| 0.5–1 | 1.01605 |

These are means over log-spaced samples, not extrema. Selected point differences
reach about -3.9% near k=0.097 h/Mpc. The sign changes with scale, so a single
amplitude rescaling cannot match the shapes.

Growth-factor ratios SYREN/JAX at z=0, 0.5, 1, 2 were respectively
1.00000, 1.00039, 1.00033, 1.00038. The inferred z=0 sigma8 values were about
0.825123 and 0.824849 (a 0.033% difference). Linear power has
`dlnP/dlnAs=1` in both models; sigma8 has `dln(sigma8)/dlnAs=0.5`.

### Nonlinear responses and marginalized errors

Mean nonlinear amplitude responses over log-spaced k=0.001–1 h/Mpc were:

| z | JAX EH + Halofit | SYREN-NEW |
| --- | ---: | ---: |
| 0 | 1.1060 | 1.0821 |
| 0.5 | 1.0921 | 1.0707 |
| 1 | 1.0803 | 1.0719 |
| 2 | 1.0590 | 1.0616 |

These averages hide scale-dependent differences; they are not pointwise accuracy
bounds. The nonlinear prescriptions also differ independently of the linear input.

For the pessimistic Euclid GCph+WL setup with ns and mnu fixed:

| Treatment of remaining parameters | sigma(10^9As), SYREN FD | JAX AD |
| --- | ---: | ---: |
| All other parameters fixed (`1/sqrt(F_AsAs)`) | 0.00191324 | 0.00186387 |
| Marginalize other cosmology; fix nuisances | 0.0223482 | 0.0237162 |
| Marginalize cosmology and ten galaxy biases; fix IA | 0.0294102 | 0.0263472 |
| Marginalize cosmology, biases and IA | 0.0294396 | 0.0264081 |

Bias marginalization changes the comparison substantially. For example,
`r(10^9As,b8)` is about -0.365 for SYREN and -0.275 for JAXcosmo. Replacing all
derivatives while holding covariance fixed nearly reproduces the corresponding
model's error, whereas swapping covariance alone has little effect in this setup.
This supports a power-response/degeneracy explanation rather than a covariance
assembly explanation.

At the second fiducial (`10^9As=2.15`, `Omegam=0.31`, `Omegab=0.049`, `h=0.68`,
same ns and mnu), the marginalized errors are 0.0307265 versus 0.0272525
(-11.31%, compared with -10.30% at the first fiducial).

### Controls and interpretation limits

| First-fiducial route | sigma(10^9As) |
| --- | ---: |
| Standard SYREN FD | 0.0294396 |
| JAX EH + Halofit AD | 0.0264081 |
| Same JAX theory and observable, FD | 0.0264065 |
| SYREN power inside the JAX observable, FD | 0.0294160 |
| Experimental Halofit on tabulated SYREN linear shape, FD | 0.0301323 |

The full SYREN power swap recovers its standard amplitude error to about 0.08%,
with globally normalized fiducial angular-spectrum difference 8.3e-4. JAX AD
versus JAX FD has maximum Fisher difference divided by the maximum reference
matrix magnitude of about 1.8e-5 at both fiducials. These are useful implementation
controls, not elementwise relative-error bounds or a general scientific accuracy
certification.

The last row suggests the linear shape is important, but **does not precisely
partition the gap into linear and nonlinear contributions**: its wide transfer
table samples a finite-range SYREN spline that clamps outside its domain and
Halofit renormalizes the shape to sigma8. Integration-tail, table-resolution,
normalization and derivative-convergence checks are needed before interpreting
the +2.35% residual as exclusively a nonlinear-prescription effect. Earlier
wording that the linear input carries “essentially all” of the gap was too strong.

The response-optimized SYREN coefficient variants did not close this fixed-ns
gap: their amplitude errors were about 0.0299734, 0.0303406 and 0.0306791 for
loss weights 0.1, 0.2 and 0.3, versus 0.0294396 for the pinned original.

**Release decision:** accept the present approximate cross-model agreement for
now, preserve these distinctions in documentation, and defer a common JAX-SYREN
theory implementation. This acceptance does not select either power model as truth.

## PyPI dependency correction and release prerequisite

**PyPI does not accept published dependency metadata containing direct URL
requirements**, including requirements placed in extras. Pip can install such
requirements from a local checkout or a URL; this does not imply PyPI accepts the
same package metadata. The earlier advice that the current GitHub archive pin
could remain in a PyPI release was incorrect.

Source: [Setuptools dependency management — Direct URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies).

Before publishing CosmicJellyfish:

- Audit runtime and optional dependency metadata, including the current
  `symbolic-pofk @ https://github.com/.../cb4d2bb...tar.gz` requirement.
- Prefer a compatible, tested package-index release with a version requirement.
  Confirm that it actually contains the SYREN-NEW APIs used by the code; do not
  assume the published version matches the pinned source.
- Keep development-only Git/worktree overrides outside published `Requires-Dist`.
- If no compatible release exists, decide on upstream release coordination or
  appropriately licensed vendoring/separate publication. Resolve this existing
  dependency issue without requiring the deferred JAX port.
- Inspect built wheel/sdist metadata and check installation from release artifacts
  in a fresh environment.

## Release and rename checklist

Status legend: `[x]` decided or done, `[ ]` open. Nothing below has been
implemented in the source tree unless explicitly ticked. Origin: planning
review of 2026-10-31 (OpenCode session `ses_e9cb32998ffeGFiwUvaXDGLfP3`),
which used `cfp-jaxcosmo` as the prospective release baseline.

### Names involved

| Identity | Proposed name | Affects |
| --- | --- | --- |
| GitHub repository | `cosmicjellyfish` | URLs, CI, badges, documentation links |
| PyPI distribution | `cosmicjellyfish` | `pip install cosmicjellyfish` |
| Python package | `cosmicjellyfish` | `import cosmicjellyfish` |
| Legacy import | `cosmicfishpie` | Existing user scripts, some saved Python objects |

About 245 tracked files contain `cosmicfishpie` (code, tests, scripts, notebooks,
docs, historical material). That is an inventory, not a list of files to change
mechanically: historical notebooks, saved outputs and provenance records should
be preserved rather than renamed or rerun indiscriminately.

### Decisions already made

- [x] Release CosmicJellyfish first; defer the JAX-SYREN fork/package work.
- [x] Accept the current roughly 10% cross-model Fisher agreement for now.
- [x] `cosmicjellyfish` is the canonical package and import name.
- [x] Keep temporary support for existing `cosmicfishpie` imports.
- [x] Rename the existing GitHub repository (keeps issues, stars, history, PRs)
      rather than creating a new one. The rename itself is done by the user.
- [x] Notebooks and scripts get tiered validation: inventory everything, execute
      maintained examples and representative workflows, schedule expensive
      scientific campaigns separately (other workstation).
- [x] JAX style work is targeted homogenization, not a wholesale rewrite.

### Open decisions

- [ ] **Legacy-import support period**: a fixed window (for example one major
      release) or until there is evidence users have migrated.
- [ ] Policy for old pickles and sampler checkpoints: promise resume, or document
      them as unsupported. Caches can generally be treated as disposable.
- [ ] Version number of the first CosmicJellyfish release.
- [x] Release baseline (decided 2026-11-02): merge `jaxcosmo-autodiff` into `main`
      through a PR and release from `main`; style homogenization (Phase 5) becomes a
      final, separate branch after that. See Phase 0b.
- [ ] **Rename timing**: does the first release go out as `cosmicfishpie` (rename
      afterwards, as a second release) or is the rename done before it? This
      decides whether Phases 1-2 come before or after Phase 3. Since
      `cosmicfishpie` was never on PyPI, renaming before the first PyPI release
      avoids ever publishing the old name; recommended unless the rename would
      delay the JAXcosmo merge too much.
- [ ] How to resolve the `symbolic-pofk` dependency (see Phase 3).

### Phase 0: release baseline

- [x] Baseline candidate recorded (2026-11-02): worktree `cfp-jaxcosmo`, branch
      `jaxcosmo-autodiff`, HEAD `1c5b946` when recorded (now `29454b6`, after the docs-mock and JAX CI commits). Confirmed by the
      user: this branch is merged into `main` via PR (Phase 0b), and the release
      is cut from `main`.
- [x] Inventory (2026-11-02): remote `origin` is
      `git@github.com:santiagocasas/cosmicfishpie.git`; the local clone has **no
      git tags**; version in `cosmicfishpie/version.py` is 1.3.1; pyproject
      `requires-python = ">=3.10,<3.13"`; workflows are `ci.yml`, `docs.yml`,
      `main.yml`, `pr_checks.yml`; other branches include `main`, `dev`,
      `cfp-docs`, `gh-pages`, `perf-v2`.
- [x] Published PyPI state: the user confirmed (2026-11-02) that **`cosmicfishpie`
      was never published to PyPI**. This matches the anonymous 404 and the absence
      of tags and GitHub releases, so the tag-triggered `main.yml` release has never
      run. Consequence: there is no PyPI user base to migrate, so the first
      PyPI release can be `cosmicjellyfish` directly. The legacy-import shim is then
      only for users who installed from GitHub or a local clone.
      Anonymous queries also returned 404 for `symbolic-pofk` while `jax-cosmo`
      returned 200; the `symbolic-pofk` result is unexpected, so it is still to
      be checked in a browser. Also check TestPyPI.
- [ ] Check that `cosmicjellyfish` is free on PyPI and TestPyPI and that the
      account that will publish owns it. A 404 is not a reservation.
- [ ] Write down the migration and compatibility policy (the open decisions above).
- [ ] Keep same-theory AD/FD checks documented separately from cross-model validation.

Gate: an agreed migration and compatibility policy.

### Phase 0b: merge `jaxcosmo-autodiff` into `main`

State on 2026-11-02 (the agent never pushes; the user pushes):

- `jaxcosmo-autodiff` was 18 commits ahead of `origin/main` and 0 behind (merge
  base `4b2bcca`), so the merge was conflict-free at that point (diffstat
  versus main then: 35 files, +5895/-34). Stale as of 2026-11-02: `main` moved
  on, so see "Merge resolution" below.
- Draft PR **#231** (base `main`, mergeable). The user has pushed the branch, so
  the PR head contains the whole JAXcosmo work (head `482143c` when last checked).
- CI on PR head `482143c`: Build, Lint, Style, Test, Docs (build only) and CHANGELOG
  pass. Only `codecov/patch` (0.40%, 991 lines missing) and `codecov/project`
  (69.59% -> 61.07%) fail.
  - Earlier failures, now fixed: the `isort` complaint was about a file that no
    longer exists; the strict docs build failed because the docs job has no jax and
    autosummary imports the jaxcosmo modules. Fix: `autodoc_mock_imports = ["jax",
    "jax_cosmo"]` in `docs/source/conf.py` (commit `482143c`), verified by
    reproducing the failure locally with jax hidden.
  - Codecov cause: the default `test` job does not install the `jaxcosmo` extra, so
    the four JAX test files skip themselves (`pytest.importorskip`) and every
    jaxcosmo module counts as 0% covered. Fix: a separate `test (jaxcosmo)` job
    (commit `29454b6`, see below).

Checklist:

- [x] Reproduce the docs failure on the current head and fix it (`482143c`).
- [x] User pushes the local commits so PR #231 reflects the real branch.
- [x] Add the JAX CI job `test (jaxcosmo)` to `.github/workflows/ci.yml`
      (`29454b6`, pushed). It installs `--extra dev --extra jaxcosmo` on Python 3.10 and
      runs the JAX test files (31 tests at that point, about 1.5 min; reproduced locally
      on Python 3.10.12 with jax 0.6.2, none skipped), then uploads coverage with the
      Codecov flag `jaxcosmo`.
- [x] User pushes `29454b6` and confirms `test (jaxcosmo)` is green on GitHub.
- [x] Re-check the Codecov statuses after the JAX upload lands. On `29454b6` the JAX job
      passes and takes `codecov/patch` from 0.40% to 67.23% (target 69.59%); project is
      69.40% versus 69.59% on `main` (-0.19%). The repository has no `codecov.yml`, so the
      Codecov defaults apply (patch must reach the base project coverage, project may not
      drop at all).
- [x] Close the remaining gap with tests rather than by moving the target: a new
      `tests/jaxcosmo_photo_parity_test.py` compares the JAX photometric observable with
      the legacy `ComputeCls` at the fiducial point, at a shifted cosmology and at
      shifted bias / IA amplitudes (agreement `max |dC|/max|C| ~2e-5` against a `1e-3`
      assertion), and checks the autodiff Jacobian against central differences. Coverage
      of `jaxcosmo_photo.py` goes from 16% to 92% (19 of 248 statements left, all error
      branches and the unused bias models), which should clear both Codecov statuses.
      The file runs in `test (jaxcosmo)`; the validation script gained a
      `_legacy_observable(calc, photobiaspars=..., iapars=...)` helper and an optional
      `fiducial` argument to `_make_calc` so the tests can reuse it.
- [ ] Confirm both Codecov statuses go green on the pushed head. Optional if the two
      uploads race: `codecov.yml` with `after_n_builds: 2`.
- [x] `CHANGELOG.md`: the JAXcosmo backend entries are already in "Unreleased"; entries
      for the JAX CI job, the docs mocking (`5dbf26c`) and the photometric parity tests
      were added.
- [x] Update the PR #231 title/description (now "Add differentiable JAXcosmo backend and
      support NumPy 2").
- [x] CodeRabbit review, round 1 (17 findings, head `07625c2`): 15 fixed in
      `f8bc762` (clustering tracer, neutrino/requested-probe validation, WL kernel at a
      zero bin edge, ordered `z_bins_sp`, name-based `Ps_*` bins, `settings["nonlinear"]`,
      derivative cache-key source, missing-spectrum reporting, derivative-provider tests,
      `FoG_switch` parity case). Deliberately skipped: the `dz_norm = z_max/1000` spacing
      is identical in the legacy `photo_window.py`, so a JAX-only fix would break parity
      and change legacy results; and the "assert a real `JaxPhotoObservable` in CI" item
      was already satisfied by `tests/jaxcosmo_photo_parity_test.py`.
- [x] CodeRabbit review, round 2 (11 findings): all fixed in `0dcdd4b` (theta-like
      expansion parameters, conflicting amplitude aliases, non-positive `sigma8`, rootless
      Halofit bracket plus a `Pnl` finiteness check, Halofit k axis, mixed/unsupported
      probe requests, `Ps_*` derivatives via the JAX Jacobian, metric wording). The new
      spectroscopic Jacobian test needs a 1e-6 relative step because the de-wiggled power
      is interpolated on a fixed k grid, so larger steps straddle interpolation kinks.
      All 22 review threads are answered and resolved; full suite 497 passed, 1 skipped.
- [x] CodeRabbit review, round 3 (12 findings, head `3458eaf`): all fixed. Provider:
      all four neutrino aliases are checked, ignored density parameterizations
      (`omega_cdm`/`omega_b`/...) raise, the amplitude must be positive and finite, and
      `NUMERICS` grids need at least four nodes for cubic splines. Theory: `distance_steps`
      is validated before truncation, and `LCDM` raises on a conflicting `w0`/`wa`
      (both in `resolve_parameters` and in direct `tables`/`power` calls). Observables:
      `_same_value` compares to `rtol=1e-12`, the spectroscopic `h_fid` comes from the
      resolved cosmology (so an `H0` fiducial is honoured), and the unimplemented
      `rescale_sigma_pv` (`sigmap_*`/`sigmav_*`) parameters raise instead of returning a
      silent zero derivative. Scripts/Halofit: `--fd-step` must be finite and non-zero,
      and a Newton bracket with no root now returns `NaN` so the provider's finiteness
      check rejects the cosmology. New regression tests cover each. Full suite 509 passed,
      1 skipped (73% coverage); isort/black/ruff and `git diff --check` clean. Committed as
      `d7d000c`, pushed, and all 12 threads answered and resolved; PR #231 has no open
      review threads at that head.
 - [x] Coverage upload (`aba5133`, CI only): Codecov briefly reported 60.50% project
      coverage with every jaxcosmo module at 0.00% and then passed minutes later without a
      code change. Cause: two uploads per run (the plain `test` job without the `jaxcosmo`
      extra, and `test (jaxcosmo)` running only the JAX files), so the first partial upload
      published a status while all JAX lines still looked uncovered. Fix: the no-jax job now
      runs `pytest --no-cov` with no upload, and `test (jaxcosmo)` runs the whole suite and
      is the only upload, so the report is never partial. Rejected `after_n_builds: 2`,
      which would wedge a status pending forever whenever one job fails or skips. Validated
       on Python 3.10 with jax 0.6.2: 507 passed, 3 skipped, 73% coverage.
 - [x] CodeRabbit review, round 4 (5 findings, head `aba5133`): all fixes and tests are in
       `9a9dc31`, pushed, and all five CodeRabbit threads have been answered and resolved.
       Fixes validate conflicting/ignored density aliases, positive finite `h`, conflicting
       `w0`/`w0_fld` and `wa`/`wa_fld`, and integral
      `NUMERICS` sample counts. The spectroscopic observable now permits unchanged unit
      `sigmap_*`/`sigmav_*` fiducials but rejects unsupported non-unit rescaling. Local
       focused suite: 38 passed; full suite: 513 passed, 1 skipped (73%); Black, isort and
       Ruff pass.
- [ ] Run the NumPy-2 backend validation `scripts/validation/run_parallel_validations.sh`
      against the stored NumPy-1 results (the PR description lists it as not yet run).
      Doctest failures in `analysis/fishconsumer.py` (3) are identical on `main`.
      Readiness audit (2026-11-03): source `9a9dc31` and cjfp-data `d6ded71` equal
      their remote heads; Fisher DVC data/cache/remote are in sync. All 25 cases
      resolve inputs; runner tests pass (3). Archive has 33 complete backend-pair
      runs and five incomplete attempts; saved gate `07.2.0.1` already fails.
      Added an uncommitted data-side preflight/isolated launcher and runbook at
      `cjfp-data/scripts/run-fisher-validation.sh` and
      `cjfp-data/validation/NUMPY2_VALIDATION.md`. Preflight passes with NumPy
      2.2.6, SciPy 1.17.0, CAMB 2.0.4, CLASS 3.3.4.0; mocked launch test passes.
      Run the PR branch before merging, then compare each backend to its own
      archived output with matched scientific inputs. Archive uses CAMB 2.0.3
      and lacks exact NumPy/SciPy versions, so observed drift requires attribution.
      Workstation checkouts and the full scientific campaign remain unverified;
      publish the data-side launcher/runbook before using them there.
- [ ] Take the number of the first release from the open decision above.
- [ ] Mark the PR ready for review, merge to `main`.

**Merge resolution (2026-11-02).** `main` had moved 22+ commits past the merge
base `4b2bcca` while PR #231 was open, including #221 (Sphinx 8.1.3, Furo
2025.12.19), #220 (Black 26 formatting plus lock refresh) and #217 (CAMB
2.0.3 -> 2.0.4), so GitHub reported the PR as `CONFLICTING`. Merged
`origin/main` (tip `40eb918`) into `jaxcosmo-autodiff` with a merge commit,
`3458eaf` "Merge origin/main into jaxcosmo-autodiff" (local only, 29 commits
ahead of `origin/jaxcosmo-autodiff`; the user pushes).

- `CHANGELOG.md` conflicted: both sides added entries at the top of
  `## Unreleased`. Kept both, with the Black 26 entry first.
- `uv.lock` conflicted and was **regenerated, not hand-merged**: took
  `origin/main`'s lockfile and ran `uv lock --python 3.10`. Result: CAMB
  2.0.4, NumPy 1.26.4 -> 2.2.6, plus the `jaxcosmo` extra (jax and jaxlib
  0.6.2 / 0.10.2 / 0.12.0.1, jax-cosmo 0.1.0, ml-dtypes 0.6.0, opt-einsum
  3.4.0), 219 packages.
- `pyproject.toml` and `.github/workflows/ci.yml` auto-merged. Checked the
  result: CAMB 2.0.4, `numpy>=2.1,<2.3`, Sphinx 8.1.3 in both extras,
  `black>=26.5.1,<27.0`, `furo==2025.12.19`. The merged `ci.yml` left the
  `test (jaxcosmo)` job on `codecov-action@v6` while main had moved to `v7`;
  bumped it to match.
- Verification on Python 3.10.12 with `--extra dev --extra jaxcosmo --extra docs`:
  `uv lock --check`, `isort --check`, `ruff check` and `git diff --check` pass;
  Black 26.10.1 reports 146 files unchanged (the merged `target-version`
  `py310` covers the new modules); full pytest **495 passed, 3 skipped** in
  4m03s; strict `sphinx-build -W --keep-going` build succeeded. Two of the
  skips are `euclidemu2` missing from the scratch environment (the `emulators`
  extra); the third is the pre-existing `fisher_derived_test.py` skip.
- `tests/reference_spectro_obs_edge_cases.py` has 5 failures, but its filename
  matches neither `test_*.py` nor `*_test.py`, so pytest never collects it in
  the default run; the file is identical to `main` modulo Black whitespace, so
  the failures are pre-existing and out of scope here.
- Remaining for this phase: the user pushes `3458eaf`; re-check
  `gh pr checks 231` (especially both Codecov statuses); run the NumPy-2
  backend validation; then mark the PR ready and merge.

Gate: PR merged, CI green on `main`, backend validation recorded.

### Phase 1: package and import migration

Inventory of the rename surface, measured 2026-11-02 on `jaxcosmo-autodiff`
(`git grep -i cosmicfishpie`):

| Area | Files mentioning the old name |
| --- | ---: |
| `scripts/` | 86 |
| `tests/` | 42 |
| `cosmicfishpie/` (the package) | 35 |
| `notebooks/` | 33 (29 `.ipynb`, about 340 occurrences) |
| `docs/` | 27 |
| root files | 11 |
| `.github/` | 7 |
| `sampler_scripts/` | 4 |

There are 303 Python `import cosmicfishpie...` / `from cosmicfishpie...` lines
across about 121 `.py` files. `scripts/validation/configs/compare_run_config.env_*`
(5 files), `sampler_scripts/*.yaml`, and `chains/` also contain the name.

Config and metadata files with the old name: `pyproject.toml`, `uv.lock`,
`MANIFEST.in`, `.coveragerc`, `pytest.ini`, `Makefile`, `CITATION.cff`,
`.github/workflows/{docs,main,pr_checks}.yml`, `.github/CONTRIBUTING.md`, issue
templates and the PR template.

Hazards for a blind find-and-replace:

- Dynamic import by string: `cosmicfishpie/configs/context.py:196`,
  `import_module("cosmicfishpie.configs.config")`. Import checkers will not see it.
  Grep for other string forms (`"cosmicfishpie.`, `'cosmicfishpie.`, paths).
- Notebooks hold saved outputs: change only the import lines in code cells, do not
  rerun them wholesale, leave outputs and old results alone.
- Old pickles store class paths such as `cosmicfishpie....` (open pickle policy).
- Keep the old name in `CHANGELOG.md`, historical docs, validation records and
  `cjfp-data` references.

Approach:

- [x] Branch `rename/cosmicjellyfish` created from `jaxcosmo-autodiff` HEAD `4b44649`
      (2026-11-03) in the worktree `cfp-rename`. This deliberately branches *before*
      PR #231 is merged, so the rename and the JAXcosmo work can be reviewed together;
      the mechanical script made the rebase risk small.
- [x] One mechanical commit planned, produced by a script with an explicit allow-list
      per file category (`/tmp/opencode/apply_rename.py`; byte-identical JSON
      round-trip verified for all 29 notebooks before running it).
- [x] Review the diff by category, not line by line.
- [ ] Add the legacy shim and its tests in a separate commit (staged, not committed).
- [x] After the rename run `isort --check`, `black --check`, `ruff check`, the
      full tests, the strict docs build, and wheel and sdist installs outside the checkout.
- [ ] Rerun a small set of maintained notebooks, not all of them.

Tasks:

- [x] Rename the canonical package directory to `cosmicjellyfish` (`git mv`).
- [x] Update internal absolute imports and dynamic or string-based imports
      (`importlib`, string module paths, config strings). 187 tracked files rewritten;
      293 files changed in total including the 89 file renames.
- [x] Update `pyproject.toml` (name, packages, package-data, version lookup),
      `uv.lock`, `.coveragerc`, `MANIFEST.in`, `pytest.ini`, entry points and
      any configuration paths that include the old name.
- [x] Update `cosmicfishpie/version.py` location (`cosmicjellyfish/version.py`) and
      every reader of the version.
- [x] Implement the legacy `cosmicfishpie` shim so that it exposes the **same
      module objects** as `cosmicjellyfish` (a `sys.meta_path` alias finder inserted
      before `PathFinder`, because the aliased parent's `__path__` points at the real
      package directory and would otherwise load a second copy).
- [x] Emit a `DeprecationWarning` from the legacy import, with migration guidance.
- [x] Preserve output names, environment variables (`COSMICFISH_FAST_*`) and
      historical records. Kept the `cosmicfishpie_version` sampler metadata key, the
      `cosmicfishpie-validation` repository name, historical `CHANGELOG.md`, the
      historical `docs/*.md` reports, the `_static/ns_response` metadata and notebook
      prose/outputs unchanged.
- [x] Tests: old and new import order, nested imports in both orders, shared
      global state, package data present, multiprocessing (manual spawn check),
      representative legacy artifact (old pickle module path) loading.
      `tests/legacy_import_test.py` (8 tests).
- [x] Tests: clean wheel and sdist installs outside the checkout
      (`/tmp/opencode/rename_install`, `/tmp/opencode/rename_sdist`): version, module
      path, package data and both import names verified.

Results (2026-11-03, `cfp-rename` worktree, Python 3.12.9, NumPy 2.2.6):

- Full suite without the JAX extra: 451 passed, 8 skipped, 60% coverage.
- Full suite with `--extra jaxcosmo`: **546 passed, 3 skipped, 73% coverage**.
- JAX files alone: 95 passed (~3 min).
- Strict `sphinx-build -W --keep-going` succeeds (before and after the jax install).
- `isort --check`, `black --check`, `ruff check`, `uv lock --check`, `git diff --check`
  all clean. Notebook diff audit: 27 notebooks changed, **zero** changes outside code
  cell sources (no outputs/metadata touched).
- `uv lock` still resolves 219 packages; the lock's project entry is renamed only.

Decisions taken inside the rename:

- `neutrino_scheme` now defaults to `"cosmicjellyfish"` but `"cosmicfishpie"` is still
  accepted, so existing YAML keeps working (covered by a parametrized test).
- The old `cosmicfishpie` import name is a pure alias: no code copy, identical module
  objects, so configuration globals, locks, class identity and pickles are shared.
- Deviations to review: `install_external_data.py` remote URLs now point at the
  `cosmicjellyfish` repository, which does not exist until Phase 2 (they were already
  stale: they point at a `survey_specifications/` directory that is not in the repo).
  `notebooks/`, `sampler_scripts/` and `chains/` are still picked up by setuptools
  namespace discovery (pre-existing packaging behaviour, unrelated to the rename).

Committed (2026-11-03) on `rename/cosmicjellyfish`, branched from `4b44649`:

- `a70abc5` Rename the package to cosmicjellyfish (292 files, 89 renames).
- `2273382` Keep the cosmicfishpie import path as a deprecated alias (shim,
  `tests/legacy_import_test.py`, `docs/source/migration_cosmicjellyfish.md`,
  index toctree entry). The 8 alias tests were re-run after the commit.

Not pushed yet. Branched from the JAXcosmo branch head rather than from `main`, so the
rename PR must either stay stacked on PR #231 or be rebased after #231 merges.

Gate: all of the above tests pass; existing numerical results unchanged.

### Phase 2: GitHub and documentation identity

- [ ] User renames the GitHub repository; then update local remotes.
- [ ] Verify that old web, clone, fetch and push URLs redirect.
- [ ] Do not reuse the old repository name later (it would break the redirect).
- [ ] Fix references to Actions by old repository name (these do not redirect).
- [ ] Audit GitHub Pages and the custom documentation domain: project-site URLs
      do not redirect automatically. Plan redirects and check old deep links.
- [ ] Update badges, repository links, `CITATION.cff`, `.all-contributorsrc`,
      `.readthedocs.yaml` and CI path filters.
- [ ] Coordinate `cjfp-data` validation references and dashboard provenance.
- [ ] Update `AGENTS.md`, `README.md` and `CHANGELOG.md`.

Gate: old repository redirects and new docs URLs verified after the rename.

### Phase 3: PyPI and TestPyPI pipeline

Current state: `.github/workflows/main.yml` is a tag-triggered production
release (build, checks, PyPI upload, GitHub release). There is no TestPyPI workflow.

- [ ] **Blocker**: replace the direct-URL `symbolic-pofk @ https://github.com/...`
      requirement. PyPI rejects direct-URL dependencies in published metadata,
      including extras. Options: a compatible index release containing the
      SYREN-NEW APIs, upstream release coordination, or licensed vendoring or
      separate publication. Verify numerical equivalence with the pinned commit
      `cb4d2bb`. Do not require the deferred JAX port.
- [ ] Fix the release job: it creates a `uv` environment but calls `python` and
      `twine` without activating it or using `uv run`.
- [ ] Verify that the tag version matches the built package version.
- [ ] Inspect built wheel and sdist metadata (`twine check`, `Requires-Dist`).
- [ ] Test installing the wheel and the sdist in fresh environments; check base
      and JAX-extra imports and package data.
- [ ] Add a separate TestPyPI workflow, preferably with PyPI Trusted Publishing,
      using a distinct prerelease version. Handle dependency resolution
      deliberately, since not every dependency is hosted on TestPyPI.
- [x] Explicit JAX-enabled CI job: drafted and committed as `29454b6` (see Phase 0b);
      verification on GitHub pending the push.
- [ ] Widen the Python matrix (3.10 only today; `requires-python` is `>=3.10,<3.13`)
      as practical, including the JAX job.
- [ ] Replace or harden `scripts/maintenance/release/release.sh`: it uses
      `git add -A` and pushes all local tags.
- [ ] Write a maintainer runbook (extend `docs/RELEASE_PROCESS.md`): TestPyPI
      setup, production release, recovery from a partial upload.
- [ ] Rehearse on TestPyPI before any production tag.

Gate: successful TestPyPI rehearsal and reproducible clean-environment installs.

### Phase 4: notebooks, scripts and documentation

- [ ] Create an index of notebooks and scripts: purpose, prerequisites, expected
      cost, outputs, and maintained versus historical.
- [ ] Notebook 05: says its Fisher inputs are committed, but no tracked
      `notebooks/results/` inputs were found; also asks users to run notebook 06
      first, contrary to numbering. Fix the dependency and the order.
- [ ] Notebook 02: uses CAMB but the output name has a `-class` suffix.
- [ ] Fix script documentation that points to paths that have moved.
- [ ] Review the site-specific Slurm wrapper: its sourced `common.sh` can override
      the submitted checkout path, which affects reproducibility.
- [ ] Leave historical and investigation notebooks with stale or cached results
      identified as such; do not rerun them indiscriminately.
- [ ] Execute maintained tutorials and representative scripts into fresh output
      locations; validate optional backends separately.
- [ ] Build the docs strictly and inspect the **rendered** site, including links to
      files outside `docs/source`.
- [ ] Reconcile installation, API, tutorials, changelog and citation.
- [ ] Write the concise user migration guide (`cosmicfishpie` to `cosmicjellyfish`).

Gate: every maintained entry has a role and a validation result.

### Phase 5: JAX style cleanup (separate small commits)

- [ ] Move shared parameter helpers out of `jaxcosmo_photo.py`; spectroscopy
      currently imports private helpers from the photo module.
- [ ] Consolidate duplicated photo/spectro Jacobian bookkeeping where it improves
      readability.
- [ ] Clarify which functions run on the host and which must accept JAX tracers.
- [ ] Correct misleading type annotations, especially spectroscopic dictionaries
      with both string and integer keys.
- [ ] Keep the small theory protocol; add no further abstraction until a second
      differentiable backend shows what is needed.
- [ ] Preserve numerical safeguards, cache invalidation, units and derivative behavior.
- [ ] Compare before and after photo and spectro observables, Jacobians, Fishers
      and cache behavior.
- [ ] Inspect the instructions of any style-review skill (for example
      `improve-codebase-architecture`, not installed; its workflow creates GitHub
      issues) before using it.

Gate: same-theory numerical parity and existing JAX tests intact; no physics
changes hidden inside style commits.

### Suggested order

Revised 2026-11-02 after the user decided to merge `jaxcosmo-autodiff` into `main`
first and to treat style homogenization as a final extra branch:

1. Phase 0 (PyPI state, policy) and the `symbolic-pofk` dependency decision.
2. Phase 0b: PR #231 into `main` (push, CI green, validation, merge).
3. Phase 3 (PyPI/TestPyPI pipeline) from `main`; Phase 4 alongside it. First
   release from `main`, after a TestPyPI rehearsal.
4. Phases 1 and 2 (rename), either before step 3 or as a second release; this is
   the open "Rename timing" decision. Doing the rename first avoids publishing
   `cosmicfishpie` and then `cosmicjellyfish`; doing it second keeps the rename
   separate from the JAXcosmo merge.
5. Phase 5 on its own branch after the release, as a separate series of small
   commits, so rename and scientific regressions stay distinguishable.

These are planning items. No source rename, publication, fork creation or upstream
PR is performed by this note.
