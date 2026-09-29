# Backend validation

The maintained backend-validation workflow compares CLASS and CAMB Fisher forecasts using the cases under `scripts/validation/configs/`. It produces per-case reports and refreshes an HTML dashboard with the latest completed results.

## Run selected cases

From the repository root, use the validation runner to choose one or more case IDs, a case group, or all configured cases:

```bash
bash scripts/validation/run_selected_validations.sh --cases 03.1.0,03.2.0
# Or run every configured case:
bash scripts/validation/run_selected_validations.sh --all
```

The runner reuses completed results when their inputs and recorded backend provenance still match. Add `--force` to rerun selected cases. See `bash scripts/validation/run_selected_validations.sh --help` for selection and execution options.

## Dashboard and reports

The validation runner refreshes `scripts/benchmark_results/dashboard/index.html` at the end of a run. To render the dashboard from existing results without rerunning forecasts:

```bash
uv run python scripts/validation/render_validation_dashboard.py
```

Add `--serve` to preview it at `http://127.0.0.1:8000/`. The dashboard summarizes cases and links to case details, solver specifications, and comparison evidence. Result files are generated locally and are gitignored.

To publish the dashboard and landing page to GitHub Pages, run `bash scripts/validation/publish_validation_dashboard.sh` from the intended source branch. This publishes to `gh-pages`; for the full procedure see [`scripts/README.md`](../../scripts/README.md#11-backend-comparisons-and-reports).

## Validation notes

- [SYREN-NEW spectral-tilt response](syren_ns_response.md) compares linear and nonlinear ns responses for CLASS+EE2, CLASS+HMcode2020 and SYREN-NEW, with reproducible figures and an analysis of the photometric Fisher differences.
- [Historical comparison with Casas et al. (2023)](validation_comparison.md) preserves an earlier investigation snapshot and its reported deviations; it is not the current case runner or a substitute for reviewing the case configuration and generated results.
- The maintained paper-validation case rationale and inputs are documented alongside [`scripts/validation/configs/`](../../scripts/validation/configs/).
- The [scripts reference](scripts_reference.md) describes the validation commands and their usage.
