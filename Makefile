.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch cosmicfishpie/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	uv run isort --check .
	uv run black --check --diff .
	uv run ruff check .
#mypy .
	CUDA_VISIBLE_DEVICES='' uv run pytest -v --color=yes --doctest-modules tests/ cosmicfishpie/

.PHONY : build
build :
	rm -rf *.egg-info/
	uv run python -m build
