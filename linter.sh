ruff check --fix .
black .
isort .
cd docs && make html
