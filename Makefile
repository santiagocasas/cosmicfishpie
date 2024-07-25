.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch cosmicfishpie/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check --diff .
	ruff check .
#mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v --color=yes --doctest-modules tests/ cosmicfishpie/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
