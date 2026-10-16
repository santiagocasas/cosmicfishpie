# Installation

`cosmicfishpie` supports Python `>=3.10,<3.13`.

## Install from PyPI

```bash
pip install cosmicfishpie
```

`camb` and `classy` (the CLASS Python wrapper) are required dependencies, so this already
gives you both solvers. If you would rather avoid them, for example on a machine without
a C compiler, the built-in `symbolic` backend (`"code": "symbolic"`) needs no solver. The
{doc}`first_forecast` uses it.

## Install from source

Clone [the repository](https://github.com/santiagocasas/cosmicfishpie) and install it
with `uv` (recommended; CI uses it) or `pip`:

```bash
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
uv sync --extra dev        # or: pip install -e ".[dev]"
```

Leave out `--extra dev` (or `[dev]`) for a lighter install without the test and
linting tools.

If you prefer conda-style environments, create one with `mamba` (or `conda`) and install
in editable mode:

```bash
mamba create -n cosmicfishpie -c conda-forge python=3.11 pip
mamba activate cosmicfishpie
git clone https://github.com/santiagocasas/cosmicfishpie.git
cd cosmicfishpie
pip install -e .
```

## Optional extras

| Extra | Adds |
| --- | --- |
| `emulators` | `euclidemu2`, for the EuclidEmulator2 nonlinear boost with CLASS |
| `docs` | Sphinx and the theme needed to build this documentation |
| `dev` | Linters, type checker, `pytest` and coverage, packaging tools |

```bash
pip install "cosmicfishpie[emulators]"   # from PyPI
uv sync --extra emulators                # from a source checkout
uv sync --extra docs --extra dev         # several extras
```

## Survey data

The survey specifications and the Euclid luminosity-ratio file (used for intrinsic
alignment) ship with the package, so no extra download is needed for the Euclid examples.

## Check the installation

```bash
python -c "import cosmicfishpie; print('cosmicfishpie import OK')"
```

Then run the {doc}`first_forecast`. From a source checkout with the `dev` extra, `uv run pytest`
runs the test suite.
