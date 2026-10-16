# Contributing

Thanks for considering contributing! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/santiagocasas/cosmicfishpie/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/santiagocasas/cosmicfishpie/issues).  Be sure to include a clear title
and description.  The description should include as much relevant information as possible.  The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see.  Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create a feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
it first on a GitHub issue.
* Check the documentation to make sure your feature does not already exist.
* Do [a quick search](https://github.com/santiagocasas/cosmicfishpie/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/cosmicfishpie.git

    or 

        git clone git@github.com:USERNAME/cosmicfishpie.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/cosmicfishpie.git, but doesn't know anything the *main* repo, [https://github.com/santiagocasas/cosmicfishpie.git](https://github.com/santiagocasas/cosmicfishpie). You can see this by running

        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/cosmicfishpie.git (fetch)
        origin https://github.com/USERNAME/cosmicfishpie.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/santiagocasas/cosmicfishpie.git](https://github.com/santiagocasas/cosmicfishpie). To do this, run the following:

        git remote add upstream https://github.com/santiagocasas/cosmicfishpie.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/cosmicfishpie.git (fetch)
        origin https://github.com/USERNAME/cosmicfishpie.git (push)
        upstream https://github.com/santiagocasas/cosmicfishpie.git (fetch)
        upstream https://github.com/santiagocasas/cosmicfishpie.git (push)

    Finally, set up the project environment (Python 3.10-3.12). We use [uv](https://docs.astral.sh/uv/) to match CI and the rest of the tooling:

        uv sync --extra dev

    This creates a `.venv` with the package installed in editable mode plus the dev dependencies (ruff, black, isort, mypy, pytest). Prefix commands with `uv run` (e.g. `uv run pytest`) or activate `.venv` directly.

    Without `uv`, the equivalent is `pip install -e .[dev]`.

    See [`AGENTS.md`](https://github.com/santiagocasas/cosmicfishpie/blob/main/AGENTS.md) in the repository root for the full list of setup, build, lint, and test commands.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/santiagocasas/cosmicfishpie.git](https://github.com/santiagocasas/cosmicfishpie), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Test your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) runs on every pull request: style (`isort`/`black` check), lint (`ruff`), the test suite (`pytest`), a package build, and a docs build-check. Run the same checks locally *before* opening a PR:

        make run-checks

    This runs `isort --check`, `black --check --diff`, `ruff check`, and `pytest` with doctests, exactly as CI does. To auto-fix formatting instead of just checking it, run `uv run isort .`, `uv run black .`, and `uv run ruff check --fix .`.

    [`pre-commit`](https://pre-commit.com/) hooks (ruff, black, isort) are configured in `.pre-commit-config.yaml`; install them once with `pre-commit install` to catch formatting issues automatically on each commit, or run them on demand with `pre-commit run --all-files`.

    `mypy` is available (`uv run mypy .`) but is not currently enforced in CI; run it locally if you're touching typed code.

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/santiagocasas/cosmicfishpie/tree/main/tests). Tests live in `tests/` and are named `*_test.py`.

    For example, if you've fixed a bug in `cosmicfishpie/LSSsurvey/photo_obs.py`, you can run the tests specific to that module with

        uv run pytest tests/photo_obs_test.py

    See [`AGENTS.md`](https://github.com/santiagocasas/cosmicfishpie/blob/main/AGENTS.md) for the exhaustive command reference (single-test invocations, doctests, environment variables, code style conventions).

    If your contribution involves additions to any public part of the API, we require that you write docstrings
    for each function, method, class, or module that you add.
    See the [Writing docstrings](#writing-docstrings) section below for details on the syntax.
    You should test to make sure the API documentation can build without errors by running

        uv sync --extra docs
        make docs

    CI and the docs publishing workflow treat warnings as errors, so you can reproduce them with
    `uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html`.

    If the build fails, it's most likely due to small formatting issues. If the error message isn't clear, feel free to comment on this in your pull request.

    And finally, please update the [CHANGELOG](https://github.com/santiagocasas/cosmicfishpie/blob/main/CHANGELOG.md) with notes on your contribution in the "Unreleased" section at the top.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/santiagocasas/cosmicfishpie/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Writing docstrings

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to build our API docs, which automatically parses all docstrings
of public classes and methods using the [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) extension.
Please refer to autodoc's documentation to learn about the docstring syntax.
