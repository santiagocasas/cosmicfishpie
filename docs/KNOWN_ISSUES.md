# Known issues and follow-ups from the docs audit

Findings from the documentation audit that were **not fixed**. Each item lists where
it is and what is wrong. Items marked (docs) are already worked around in the docs.

## Source code

### `cosmicfishpie/configs/config.py`: docstring mismatches
- The `code` option docstring lists only `'camb'`, `'class'` and `'external'`; it omits
  `'symbolic'`.
- `bfs8terms` is documented with default `True` but is set to `False`.
- The option is named `fix_cosmo_nl_terms`, but the docstring says `fix_cosmo_nl`.
- `activateMG` is documented as a bool, but the code only acts on `"late-time"` and
  `"external"`.
- `memorize_cosmo` is documented as general, but only the external backend uses it
  (`cosmology.py` near line 1718).
- `Rubin` and `DESI` have survey-name branches, but no spec YAML is shipped for either.
  `surveyName="DESI"` fails with a missing-file error (docs).
- Only `Planck.yaml` is ever loaded by `surveyName`. The Simons Observatory and
  CMB-Stage4 YAMLs are never selected that way; pass them via `--spec-yaml` or
  `specifications` (docs).

### `cosmicfishpie/utilities/utils.py`: `load_fisher_from_json`
- The docstring says the file is `*_specifications.json`, but the file written is
  `*_FM_specs.json` (line ~169).

### `cosmicfishpie/likelihood/spectro_like.py`: inconsistent keywords
- `SpectroLikelihood` takes `cosmoFM_data` and `cosmoFM_theory`. `PhotometricLikelihood`
  and the base class take `cosmo_data` and `cosmo_theory`. Accepting both names in
  `SpectroLikelihood` would remove the trap (docs).

### `cosmicfishpie/configs/install_external_data.py`: dead URLs
- The four `raw.githubusercontent.com/.../survey_specifications/*` downloads return 404.
  `lumratio_file.dat` already ships with the package. The docs no longer tell users to
  run the script.

### `cosmicfishpie/fishermatrix/derivatives.py`
- `derivative_poly` and `derivative_stem` still fail for non-photometric probes, and
  `4PT_FWD` still fails for CMB probes. Only a `UserWarning` was added. Details are in
  the "Observable support by method" table in `docs/source/derivatives.md`.
- `DerivativeProvider` is a `Protocol` but not `@runtime_checkable`, so
  `isinstance(x, DerivativeProvider)` raises.
- A custom provider that ignores `special_deriv_function` silently drops analytic
  overrides.
- The `"plain"` observable type with `3PT` needs an array-valued observable; a
  dict-valued one fails with `TypeError`.

### `cosmicfishpie/likelihood/`
- `tests/likelihood_base_test.py` mostly wraps checks in `try/except` around names that
  do not exist (such as `BaseLikelihood`), so it asserts almost nothing.
- `plot_chain_summary` has no `FishConsumer` method (module-level function only).

### `pyproject.toml`
- The `dev` extra repeats the Sphinx pins from `docs` with a looser `Sphinx>=4.3.0` bound.

## Scripts

- `scripts/validation/setup/install_camb_fork_for_validation.sh`: the `--help` text refers
  to `scripts/install_camb_fork_for_validation.sh`; the real path is
  `scripts/validation/setup/install_camb_fork_for_validation.sh`.
- `ns_response_metadata.json` records `k_hmpc.samples: 243`, while
  `plot_ns_power_response.py` defaults `--k-samples` to 240. The artifact was probably
  made with non-default flags that the metadata does not record.
- `docs/RELEASE_PROCESS.md`, "Fixing a failed release": it deletes all local tags
  (`git tag -l | xargs git tag -d`), not just the broken one. Not wrong, but blunt.

## Tests missing

- No unit test for the new `survey_name_radio_IM` default (`SKAO-IM-Redbook`) and the
  `other_survey_specifications` lookup. It was checked manually with
  `surveyName="SKAO"`, `observables=["IM"]` and the symbolic backend.

## Docs

- The `kmax_IM` values in the survey-file table are given without units; they could not
  be confirmed from the code.
- The survey-table numbers (`lmax`, `fsky`, `kmax`, deg²) were not re-checked against
  the YAML files after the guide was condensed.
- `likelihood_sampling.md` has no custom-component code example after trimming.
- `docs/source/index.md` does not yet mention likelihood sampling in the User guide card
  or the Features list.
- The `external` backend was tested by loading the data and reading `H(0.5)`, but not
  with a full `FisherMatrix.compute()`. A custom `derivative_provider` was not run
  through a full `FisherMatrix` either.
- `SpectroLikelihood` sampling was not run end to end (only the photometric example and
  the WL `NautilusSampler` config were).

## Decisions for the maintainer

- Whether to fix the source issues above in a follow-up PR.
- After the first merge to `main`: check that `docs/` appears on `gh-pages` and that
  `dashboard/` is untouched.
- A live per-PR docs preview on `gh-pages` is not implemented. PRs get a `docs-preview`
  artifact instead.
