# Agent Guide for CosmicFishPie

This file is for coding agents working in this repository.
Follow existing patterns and keep changes minimal and focused.

## Scope and intent
- Repository: scientific Python package for cosmology Fisher forecasts.
- Primary languages: Python 3.10-3.12 (CI uses 3.10).
- Use uv-based workflows when available to match CI and tooling.

## Quick setup
- Install dependencies (recommended): `uv sync --extra dev`
- Alternative editable install: `pip install -e .[dev]`
- Confirm environment: `uv run python --version`

## Build, lint, and test commands
- Run all checks (format, lint, tests): `make run-checks`
- Format imports: `uv run isort .`
- Format code: `uv run black .`
- Lint (no fixes): `uv run ruff check .`
- Lint (apply fixes): `uv run ruff check --fix .`
- Type check: `uv run mypy .`
- Run tests (CI baseline): `uv run pytest`
- Run tests + doctests (local full): `CUDA_VISIBLE_DEVICES='' uv run pytest -v --color=yes --doctest-modules tests/ cosmicfishpie/`
- Build package: `make build` (or `uv run python -m build`)

## Run a single test
- Single file: `uv run pytest tests/photo_obs_test.py`
- Single test: `uv run pytest tests/photo_obs_test.py::test_name`
- By pattern: `uv run pytest -k "photo and not slow"`
- Tests live in `tests/` and are typically named `*_test.py`.

## Docs
- Live docs build (autobuild): `make docs`
- Standard Sphinx build: `make -C docs html`
- API docs are built from module docstrings; keep them clean and complete.

## Repository layout
- `cosmicfishpie/` is the main package.
- `cosmicfishpie/analysis/` contains Fisher post-processing and plotting.
- `cosmicfishpie/cosmology/` handles cosmology calculations and nuisance models.
- `cosmicfishpie/fishermatrix/` contains Fisher matrix computation and derivatives.
- `cosmicfishpie/LSSsurvey/` and `cosmicfishpie/CMBsurvey/` hold probe-specific logic.
- `cosmicfishpie/utilities/` has shared helpers (printing, numerics, filesystem).
- `cosmicfishpie/configs/` stores YAML specs and external data definitions.
- `scripts/` holds runnable workflows and benchmarks (not imported by the library).
- `tests/` contains pytest tests and fixtures.

## Configuration and inputs
- Global options live in `cosmicfishpie.configs.config` and `cfg.settings`.
- Prefer adding new survey and solver configs under `cosmicfishpie/configs/`.
- Keep YAML inputs in `configs/default_*` or `configs/other_*` folders.
- Avoid hard-coded paths; use config defaults (e.g., specs_dir, results_dir).
- When adding new external datasets, update configs and tests accordingly.

## Backends and optional dependencies
- Supported backends: `camb`, `class`, and `symbolic`.
- Import optional backends lazily inside functions or classes.
- If a backend is missing, raise `ImportError` or exit with guidance in scripts.
- Fast-path toggles are controlled by env vars:
  `COSMICFISH_FAST_EFF`, `COSMICFISH_FAST_P`, `COSMICFISH_FAST_KERNEL`.
- Keep backend-specific behavior isolated and covered by tests when possible.

## Code style and conventions
- Formatting is enforced by Black (line length 100). Keep new code under 100 chars.
- Ruff uses line length 115; Black is the formatter of record.
- Import order: stdlib, third-party, local (`cosmicfishpie.*`).
- isort profile is "black"; do not hand-tune import ordering.
- Prefer absolute imports within the package (see existing modules).
- Prefer `from __future__ import annotations` in new modules with type hints.
- Type hints: add for new public APIs and complex functions.
- Use `dict[str, T]`, `list[T]`, `tuple[T, ...]` (Python 3.10+).
- Avoid `Any` unless unavoidable; use `Optional` or unions explicitly.
- Naming: use snake_case for functions/variables and CapWords for classes.
- Legacy modules use snake_case class names (e.g., `fisher_matrix`); match file style.
- Constants are UPPER_CASE; module-level config is in `cosmicfishpie.configs.config`.
- Prefer small, composable functions and avoid large monoliths when adding new code.
- Keep public APIs stable; avoid renaming exported symbols without deprecation.

## Docstrings
- Public APIs should have docstrings (see `.github/CONTRIBUTING.md`).
- NumPy-style docstrings are common in scientific modules; follow the local file style.
- Include units and assumptions for cosmology calculations when relevant.
- Keep docstrings Sphinx-friendly; avoid heavy markup.
- When extending older modules that use reST param style, match that style.
- Keep examples minimal and deterministic (avoid network or large data).

## Error handling and warnings
- Prefer explicit exceptions with clear messages over silent failures.
- Use `ValueError` for bad inputs and `TypeError` for type mismatches.
- Avoid bare `except`; catch specific exceptions.
- Optional dependency failures should raise `ImportError` with guidance.
- For scripts, `print` + `sys.exit()` is acceptable when user-facing.
- Use `warnings.warn` or `utilities.utils.printing.suppress_warnings` as needed.
- Do not swallow numerical warnings unless `SUPPRESS_WARNINGS` is set.

## Logging and verbosity
- Global verbosity is controlled by `cfg.settings["feedback"]`.
- Use `utilities.utils.printing.debug_print` for debug-only output.
- Use `utilities.utils.printing.time_print` for timing messages.
- Avoid noisy prints in library code unless gated by feedback level.

## Performance and numerics
- Prefer NumPy/SciPy vectorization over Python loops when possible.
- Respect configuration flags in `cfg.settings` instead of hard-coding behavior.
- Be mindful of caching directories (`memory_cache`, `cache`) and large outputs.
- Avoid changing default numerical recipes without tests and justification.
- Keep array shapes explicit; document assumptions about bins and units.
- Favor stable algorithms over micro-optimizations unless benchmarks justify changes.

## Outputs and caching
- Default outputs are written to `results_dir` from config.
- Do not commit results, plots, caches, or benchmark outputs.
- Common transient paths include `results/`, `cache/`, `memory_cache/`, `chains/`.
- Scripts often create `scripts/benchmark_results/` and `chains/` folders.

## Scripts and benchmarks
- `scripts/photometric_benchmark.py` runs photometric benchmarks and FAST vs SLOW checks.
- `scripts/run_fisher_compare_backends.py` compares Fisher matrices across backends.
- Scripts are not imported by the library; keep them CLI-friendly.
- Prefer `uv run python scripts/<name>.py` to ensure the right environment.
- Benchmark outputs are stored under `scripts/benchmark_results/`.

## Environment variables
- `COSMICFISH_FAST_EFF`, `COSMICFISH_FAST_P`, `COSMICFISH_FAST_KERNEL` toggle fast paths.
- `OMP_NUM_THREADS` can cap BLAS/OpenMP threading for reproducible timings.
- Use `CUDA_VISIBLE_DEVICES=''` for CPU-only test runs when needed.

## Tests and data hygiene
- Add or update tests for new behavior in `tests/`.
- Do not commit generated outputs (results, caches, benchmarks) unless requested.
- Keep configuration and YAML inputs under `cosmicfishpie/configs/`.
- Use `pytest` fixtures in `tests/conftest.py` when adding shared setup.
- Keep tests deterministic; avoid reliance on external services.
- If a test needs optional backends, mark or guard it accordingly.

## Pre-commit and CI
- Pre-commit hooks run Ruff, Black, and isort (see `.pre-commit-config.yaml`).
- CI uses `uv sync --extra dev` then `uv run pytest`.
- Prefer matching CI commands when validating changes locally.
- You can run `pre-commit run --all-files` if pre-commit is installed.

## Versioning and metadata
- Package version lives in `cosmicfishpie/version.py`.
- Citation metadata is in `CITATION.cff`.
- Update `CHANGELOG.md` for user-facing changes.

## Cursor and Copilot rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
