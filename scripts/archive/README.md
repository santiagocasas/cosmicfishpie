# Archived validation tooling

This directory preserves superseded validation scripts and inputs for historical
provenance. The maintained workflow is:

```bash
bash scripts/run_selected_validations.sh --all --omp-threads 8
```

Archived contents are not discovered by the validation dashboard and are not run by
the maintained selected-case runner.

- `run_all_validations.sh` and `validate_many_compare_configs.sh` are superseded batch
  orchestrators.
- `run_case01_full7_*` and `run_fisher_compare_w0wa_fast.py` are one-off CAMB-fork
  investigation tools. Their paths were adjusted so they remain inspectable/runnable.
- `validation_configs/` contains former cases 03-06 and their common specifications.
- `legacy_yamls/` contains solver inputs used only by archived investigations.
- `orphan_yamls/` contains package YAMLs with no live code, test, or configuration
  references at the time of archival.

Do not use these files for new validation results without first reviewing them against
the current package and dependency versions.
